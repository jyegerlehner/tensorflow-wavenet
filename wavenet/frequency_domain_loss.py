import math
import numpy as np
import tensorflow as tf

from .ops import mu_law_encode, mu_law_decode



def sample_gumbel(shape, eps=1e-10):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1-eps, dtype=tf.float32)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    noise = sample_gumbel(tf.shape(logits))
    y = tf.cast(logits, tf.float32) + tf.cast(noise, tf.float32)
    return tf.nn.softmax(y / temperature)

def gumbel_softmax(logits, temperature, hard=False, gumbel_noise=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, gumbel_noise)
    if hard:
      k = tf.shape(logits)[-1]
      #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
      y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
      y = tf.stop_gradient(y_hard - y) + y
    return y

'''
Convert the raw network output into probability distribution over the
quantization levels.

Args:
    raw_output: The output of the WaveNet, which are the logits to be used
                in the softmax producting the discrete prob distribution
                over quantization levels. Shape: W x C where W is samples
                and C is quantization levels.

    return: discrete probability distrutions for samples.
'''
def output_to_probs(raw_output, temp=1.0, use_gumbel=False,
                    gumbel_noise=False):
#    g = tf.get_default_graph()
#    with g.gradient_override_map({"Softmax": "JYCustomSoftmax"}):
    if use_gumbel:
        return gumbel_softmax(logits=raw_output,
                          temperature=temp,
                          gumbel_noise=gumbel_noise, name='probs')
    else:
        return tf.nn.softmax(logits=raw_output, name='probs')


'''
Compute the entropy of each of the discrete probability distributions,
then average them.

Args:
    probs: Tensor of probability distributions output from the WaveNet during
           training. One prob distribution for each time step.
           Each distribution is over the possible mu-law-encoded discrete
           values. Shape: W x C, where  W is width in samples
           (duration), and C is number of quantization levels.


    return: average of the entropies, in nats.

'''
def probs_to_entropy(probs):
    entropies = tf.reduce_sum(tf.multiply(probs, tf.log(probs)), axis=1)
    return -tf.reduce_mean(entropies)


'''
Compute the entropy of each of the discrete probability distributions,
then average them.

Args:
    probs: Tensor of probability distributions output from the WaveNet during
           training. One prob distribution for each time step.
           Each distribution is over the possible mu-law-encoded discrete
           values. Shape: W x C, where  W is width in samples
           (duration), and C is number of quantization levels.


    return: average of the entropies, in bits.

'''
def probs_to_entropy_bits(probs):
    return probs_to_entropy(probs) / math.log(2.0)


class FrequencyDomainLoss:
    '''
    Object for creating a loss. Loss is computed as l2 loss between tensors
    produced by filtering the target and actual outputs through a bank of
    periodic filters.

    Args:
        max_period: The period, in samples, of the lowest frequency (highest
        period) filter.

    '''
    def __init__(self, max_period, quantization_levels, dtype=tf.float32):
        # Highest frequency is that allowed by sample rate and
        # nyquist criterion.
        min_period = 2.0
        # Time to get from f0 to f1: twice the period of the lowest frequency.
        T = max_period * 10.0 #2.0
        times = np.arange(0.0, T, 1.0)
        # Start at low frequency
        f0 = 1.0/max_period
        # End at high frequency
        f1 = 1.0/min_period
        k = (f1 - f0) / T
        sine_arg = 2.0*np.pi*(times*f0 + 0.5*k*times*times)
        chirp = np.sin(sine_arg + np.pi/2.0)
        reverse_chirp = chirp[::-1].copy()
        dc_component = np.array([1.0 for time in times])
        filter_shape = [ times.shape[0], 1, 3]
        filter = np.zeros(shape=filter_shape, dtype=np.float)
        filter[:, 0, 0] = chirp
        filter[:, 0, 1] = reverse_chirp
        filter[:, 0, 2] = dc_component

        self.filter_duration = T

        # Turn the numpy array into a constant tensor.
        self.filter = tf.constant(value=filter, dtype=dtype,
                                  shape=filter_shape,
                                  verify_shape=True)

        levels = [i for i in range(quantization_levels)]
        levels_tensor = tf.constant(value=levels,
                                    shape=[quantization_levels],
                                    verify_shape=True)
        self.mu_law_values = mu_law_decode(levels_tensor, quantization_levels)
        self.mu_law_values = tf.reshape(self.mu_law_values,
                                    shape=[quantization_levels, 1])

    '''
    Convert the tensor of discrete probability distributions to expected value
    of the waveform at each time step.

    Args:
        probs: Tensor of probability distributions output from the WaveNet during
               training. One prob distribution for each time step.
               Each distribution is over the possible mu-law-encoded discrete
               values. Shape: W x C, where  W is width in samples
               (duration), and C is number of quantization levels.

        return: Tensor with the expected value of the signal amplitude.
                Shape: W x 1

    '''
    #_mu_law_values = None
    def probs_to_waveform(self, probs):
        # Expected value via total expectation theorem.
        expected_value = tf.matmul(probs, self.mu_law_values)
        return expected_value

    '''
    Compute the loss by convolving filter with the two target and actual
    tensors.

    Args:
        target: The target waveform. N x W x C where:
            N: batch
            W: Width in samples
            C: Channels == 1

        actual: The actual waveform.  N x W x C where:
            N: batch
            W: Width in samples
            C: Channels == 1

        return: The tensor of losses.
    '''
    def __call__(self, target, actual):
        with tf.name_scope('frequency_domain_loss'):
            # Expect N x W x C tensors.
            assert 3 == len(actual.get_shape())
            assert 3 == len(target.get_shape())
            a = tf.nn.convolution(input=actual,
                                  filter=self.filter,
                                  padding='SAME',
                                  data_format='NWC')
            b = tf.nn.convolution(input=target,
                                  filter=self.filter,
                                  padding='SAME',
                                  data_format='NWC')
            return tf.sqrt(tf.nn.l2_loss(a-b) / self.filter_duration)

