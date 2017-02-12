from __future__ import print_function

import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from wavenet import (output_to_probs, mu_law_decode, mu_law_encode,
                     FrequencyDomainLoss, probs_to_entropy,
                     probs_to_entropy_bits)


class TestFrequencyDomainLoss(tf.test.TestCase):


    def testOutputToProbs(self):
        # 2 time steps, 3 prob channels.
        out_array = np.zeros(shape=[2,3])
        out_array[0,:] = np.array([1.0, -2.0, 0.0])
        out_array[1,:] = np.array([0.0, 0.0, 0.0])

        expected_probs = np.copy(out_array)
        expected_probs[0,:] = np.exp(out_array[0,:]) / \
                                np.sum(np.exp(out_array[0,:]))
        expected_probs[1,:] = np.exp(out_array[1,:]) / \
                                np.sum(np.exp(out_array[1,:]))

        out_tensor = tf.constant(value=out_array,
                                 shape=[2,3],
                                 verify_shape=True)

        probs_op = output_to_probs(out_tensor, gumbel_noise=False)

        with self.test_session() as sess:
            probs = sess.run(probs_op)

        self.assertAllClose(probs, expected_probs)


    def test_probs_to_waveform(self):
        QUANTIZATION_LEVELS=256
        levels = [i for i in range(QUANTIZATION_LEVELS)]
        levels_tensor = tf.constant(value=levels,
                                    shape=[QUANTIZATION_LEVELS],
                                    verify_shape=True, name="levels")

        # This is all the possible values that can be mu-law-encoded. One for
        # each quantization level.
        amplitudes_tensor = mu_law_decode(levels_tensor, QUANTIZATION_LEVELS)

        # Waveform probs
        probs = np.zeros(shape=[3,256], dtype=np.float32)
        SAMPLE1 = 200
        SAMPLE2 = 40
        SAMPLE3 = 128
        probs[0,SAMPLE1] = 0.9
        probs[0,SAMPLE1+1] = 0.1
        probs[1,SAMPLE2] = 0.1
        probs[1,SAMPLE2+1] = 0.9
        probs[2,SAMPLE3] = 0.9
        probs[2,SAMPLE3+1] = 0.1
        probs_tensor = tf.constant(value=probs)

        loss = FrequencyDomainLoss(max_period=100,
                                   quantization_levels = QUANTIZATION_LEVELS)
        waveform_tensor = loss.probs_to_waveform(probs_tensor)
        with self.test_session() as sess:
            (waveform, amplitudes) = sess.run([waveform_tensor,
                                               amplitudes_tensor])

        expected_waveform = [amplitudes[SAMPLE1], amplitudes[SAMPLE2],
                              amplitudes[SAMPLE3]]

        self.assertGreaterEqual(waveform[0], amplitudes[SAMPLE1])
        self.assertLessEqual(waveform[0], amplitudes[SAMPLE1+1])

        self.assertGreaterEqual(waveform[1], amplitudes[SAMPLE2])
        self.assertLessEqual(waveform[1], amplitudes[SAMPLE2+1])

        self.assertGreaterEqual(waveform[2], amplitudes[SAMPLE3])
        self.assertLessEqual(waveform[2], amplitudes[SAMPLE3+1])

    def test_loss(self):
        QUANTIZATION_CHANNELS = 256
        SAMPLE_RATE = 1000
        SAMPLE_PERIOD = 1.0 / SAMPLE_RATE
        FREQ1 = 50.0  # Hz
        FREQ2 = 100.0  # Hz
        LOWEST_FREQUENCY_OF_LOSS = 10
        LOW_FREQ_PERIOD = SAMPLE_RATE // LOWEST_FREQUENCY_OF_LOSS
        loss_functor = FrequencyDomainLoss(
            max_period=LOW_FREQ_PERIOD,
            quantization_levels = QUANTIZATION_CHANNELS)

        def CreateSine(times, mag_and_freqs):
            waveform = [0.0 for time in times]
            for (mag, freq) in mag_and_freqs:
                waveform +=  mag * np.sin(2.0 * np.pi * freq * times)

            sine_tensor = tf.constant(waveform, dtype=tf.float32)
            sine_tensor = mu_law_encode(sine_tensor, QUANTIZATION_CHANNELS)
            sine_tensor = mu_law_decode(sine_tensor, QUANTIZATION_CHANNELS)

            encoded_sine = mu_law_encode(sine_tensor, QUANTIZATION_CHANNELS)
            output_sine = 10.0 * tf.one_hot(
                                    encoded_sine,
                                    QUANTIZATION_CHANNELS,
                                    dtype=tf.float32)
            sine_probs = output_to_probs(output_sine)
            sine_expected_vals = loss_functor.probs_to_waveform(
                  probs=sine_probs)

            sine_samples = sine_tensor.get_shape()[0]
            sine_reshaped = tf.reshape(sine_tensor,[1, int(sine_samples), 1])
            sine_expected_vals = tf.reshape(
                 sine_expected_vals,
                 shape=[1,
                 int(sine_expected_vals.get_shape()[0]),
                 int(sine_expected_vals.get_shape()[1])])

            return (sine_expected_vals, sine_reshaped)

        times = np.arange(0.0, 4.0, SAMPLE_PERIOD)
        (sine1_expected_vals, sine1) = CreateSine(times, [(1.0, FREQ1)])
        (sine2_expected_vals, sine2) = CreateSine(times, [(1.0, FREQ2)])
        (mixed_sine_expected_vals, mixed_sine) = CreateSine(times,
                                                    [(0.4, FREQ1),
                                                     (0.6, FREQ2)])

        loss1_1 = loss_functor(sine1, sine1_expected_vals)
        loss2_2 = loss_functor(sine2, sine2_expected_vals)
        loss2_1 = loss_functor(sine1, sine2_expected_vals)
        loss1_M = loss_functor(mixed_sine, sine1_expected_vals)
        loss2_M = loss_functor(mixed_sine, sine2_expected_vals)
        lossM_M = loss_functor(mixed_sine, mixed_sine_expected_vals)
        ops = [loss1_1, loss2_2, loss2_1, loss1_M, loss2_M, lossM_M]

        with self.test_session() as sess:
            (loss1_1_val,loss2_2_val, loss2_1_val, loss1_M_val, loss2_M_val,
             lossM_M_val) = sess.run(ops)

        print("loss1_1:{} loss2_2:{}, loss2_1:{} loss1_M:{} loss2_M:{}"
              "lossM_M:{}".format(
              loss1_1_val, loss2_2_val, loss2_1_val, loss1_M_val,
              loss2_M_val, lossM_M_val))

        # Loss of a sine wave against itself should be nearly zero, and
        # definitely smaller than the loss of the wave against a mixture of
        # itself and something else.
        self.assertLess( loss1_1_val, loss1_M_val)
        self.assertLess( loss2_2_val, loss1_M_val)
        self.assertLess( lossM_M_val, loss1_M_val)
        # Loss of a sine wave against something mixed with itself should be
        # less than the loss against a sine wave of a completely different
        # frequency.
        self.assertLess( loss1_M_val, loss2_1_val)
        self.assertLess( loss2_M_val, loss2_1_val)


    def test_entropy(self):
        QUANTIZATION_LEVELS=256
        # Uniform probs should have entropy = 8 bits.
        uniform_probs = np.array([1.0/QUANTIZATION_LEVELS
                          for i in range(QUANTIZATION_LEVELS)])
        uniform_probs = np.array([1.0/QUANTIZATION_LEVELS
                                  for i in range(QUANTIZATION_LEVELS)])

        # Probs should have zero entropy.
        # zero_ent_probs = np.zeros([QUANTIZATION_LEVELS])
        zero_ent_probs = np.array([1e-10 for i in range(QUANTIZATION_LEVELS)])
        zero_ent_probs[12] = 1.0

        probs = np.zeros([2,QUANTIZATION_LEVELS])
        probs[0,:] = uniform_probs
        probs[1,:] = zero_ent_probs

        probs_tensor = tf.constant(value=probs)

        entropy = probs_to_entropy_bits(probs)

        with self.test_session() as sess:
            entropy_val = sess.run(entropy)

#        # Convert nats to bits.
#        entropy_val = entropy_val / math.log(2.0)

        # Entropy of one distribution is 8 bits, entropy of the other is
        # 0. Mean of the two distributions = (8 + 0)/2 = 4 bits.
        self.assertNear(entropy_val, 4.0, 1e-3)


if __name__ == '__main__':
    tf.test.main()
