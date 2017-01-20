from __future__ import division

import tensorflow as tf

MIN_SAMPLE_DENSITY = 500.0
MAX_SAMPLE_DENSITY = 2000.0
DENSITY_SPAN = MAX_SAMPLE_DENSITY - MIN_SAMPLE_DENSITY


def show_params(params, indent=""):
    for key in params.keys():
        if isinstance(params[key], dict):
            print(indent+key)
            show_params(params[key], indent+"    ")
        else:
            print(indent+"    {}:{}".format(key, params[key]))


def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable

def create_embedding_table(name, shape):
    if shape[0] == shape[1]:
        # Make a one-hot encoding as the initial value.
        initial_val = np.identity(n=shape[0], dtype=np.float32)
        return tf.Variable(initial_val, name=name)
    else:
        initializer = tf.truncated_normal(shape, mean=0.0, stddev=0.3,
                                dtype=tf.float32)
        variable = tf.Variable(initializer, name=name)
        return variable

def create_bias_variable(name, shape, value=0.0):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name=name)

def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-3)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


def clamp(val, min, max):
    val = tf.maximum(val, min)
    val = tf.minimum(val, max)
    return val


def shape_size(shape):
    size=1
    for dim_size in shape.as_list():
        size *= dim_size
    return size


def gated_residual_layer(input, layer_name):
    with tf.variable_scope(layer_name):
        input_size = shape_size(input.get_shape())
        weights_shape = [input_size, input_size]
        bias_shape = [input_size]
        filt_weights = create_variable('filt_weights', weights_shape)
        gate_weights = create_variable('gate_weights', weights_shape)
        filt_bias = create_bias_variable('filt_bias', bias_shape)
        gate_bias = create_bias_variable('gate_bias', bias_shape)
        t1 = tf.matmul(input, filt_weights) + filt_bias
        t2 = tf.matmul(input, gate_weights) + gate_bias
        # Residual output.
        output = input + t1*t2
        return output

def quantize_value(value, quant_levels, min, max):
    assert max > min
    assert quant_levels > 1
    value = clamp(value, min, max)
    ratio = (value - min) / (max - min)
    quant = tf.cast(tf.floor(ratio*quant_levels), dtype=tf.int32)
    return quant


def quantize_sample_density(density, quant_levels):
    return quantize_value(density, quant_levels, MIN_SAMPLE_DENSITY,
                          MAX_SAMPLE_DENSITY)


def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation, name='causal_conv'):
    with tf.name_scope(name):
        # Pad beforehand to preserve causality.
        filter_width = tf.shape(filter_)[0]
        padding = [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]]
        padded = tf.pad(value, padding)
        if dilation > 1:
            transformed = time_to_batch(padded, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='SAME')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(padded, filter_, stride=1, padding='SAME')
        # Remove excess elements at the end.
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, tf.shape(value)[1], -1])
        return result


def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = quantization_channels - 1
        # Perform mu-law companding transformation (ITU-T, 1988).
        magnitude = tf.log(1 + mu * tf.abs(audio)) / tf.log(1. + mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.cast((signal + 1) / 2 * mu + 0.5, tf.int32)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        casted = tf.cast(output, tf.float32)
        signal = 2 * (casted / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude

