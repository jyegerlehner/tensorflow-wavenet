import numpy as np
import tensorflow as tf

CHARACTER_CARDINALITY = 256
LAYER_COUNT = 5
FILTER_WIDTH = 2
UPSAMPLE_RATE = 1000  # Typical number of audio samples per


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
        return create_variable(name, shape)


def create_bias_variable(name, shape, value=0.0):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


class ConvNetModel(object):
    def __init__(self,
                 batch_size,
                 encoder_channels,
                 histograms,
                 output_channels,
                 local_condition_channels):
        self.batch_size = batch_size
        self.encoder_channels = encoder_channels
        self.histograms = histograms
        self.output_channels = output_channels
        self.local_condition_channels = local_condition_channels

        self.variables = self._create_variables()

    def _create_variables(self):
        var = dict()
        with tf.variable_scope('encoder_convnet'):
            with tf.variable_scope('embeddings'):
                layer = dict()
                layer['text_embedding'] = create_embedding_table(
                    'text_embedding',
                    [CHARACTER_CARDINALITY, self.encoder_channels])
                var['embeddings'] = layer
            var['layer_stack'] = []
            with tf.variable_scope('layer_stack'):
                for i in range(LAYER_COUNT):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        current['filter'] = create_variable(
                            'filter',
                            [FILTER_WIDTH,
                             self.encoder_channels,
                             self.encoder_channels])
                        current['gate'] = create_variable(
                            'gate',
                            [FILTER_WIDTH,
                             self.encoder_channels,
                             self.encoder_channels])
                        current['dense'] = create_variable(
                            'dense',
                            [1,
                             self.encoder_channels,
                             self.encoder_channels])
                        current['skip'] = create_variable(
                            'skip',
                            [1,
                             self.encoder_channels,
                             self.output_channels])
                        current['filter_bias'] = create_bias_variable(
                            'filter_bias',
                            [self.encoder_channels])
                        current['gate_bias'] = create_bias_variable(
                            'gate_bias',
                            [self.encoder_channels])
                        current['dense_bias'] = create_bias_variable(
                            'dense_bias',
                            [self.encoder_channels])

            with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess1'] = create_variable(
                    'postprocess1',
                    [1, self.output_channels, self.output_channels])
                current['postprocess2'] = create_variable(
                    'postprocess2',
                    [1, self.output_channels, self.output_channels])
                current['postprocess1_bias'] = create_bias_variable(
                    'postprocess1_bias', self.output_channels, value=-0.2)
                current['postprocess2_bias'] = create_bias_variable(
                    'postprocess2_bias', self.coutput_channels, value=-0.2)
                var['postprocessing'] = current

            with tf.variable_scope('upsampling'):
                # filter for tf.nn.conv2d_transpose, with height = 1 to achieve
                # a 1d deconv.
                variables['upsampling']['filter'] = create_variable(
                    'upsamp_filter',
                    [1,
                     UPSAMPLE_RATE,
                     self.output_channels,
                     self.local_condition_channels])
        return var

    def _create_layer(input, layer_index):
        variables = self.variables['layer_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']
        filter_bias = variables['filter_bias']
        gate_bias = variables['gate_bias']

        conv_filter = tf.nn.conv1d(global_condition_batch,
                                   weights_filter,
                                   stride=2,
                                   padding="VALID",
                                   name="filter") + filter_bias

        conv_gate = tf.nn.conv1d(global_condition_batch,
                                 weights_gate,
                                 stride=2,
                                 padding="VALID",
                                 name="gate") + gate_bias

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        dense_bias = variables['dense_bias']
        transformed = tf.nn.conv1d(out, weights_dense, stride=1,
                                   padding="SAME", name="dense")
        transformed += dense_bias

        # The 1x1 conv to produce the skip output
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
                out, weights_skip, stride=1, padding="SAME", name="skip")

        if self.histograms:
            layer = 'layer{}'.format(layer_index)
            tf.histogram_summary(layer + '_filter', weights_filter)
            tf.histogram_summary(layer + '_gate', weights_gate)
            tf.histogram_summary(layer + '_dense', weights_dense)
            tf.histogram_summary(layer + '_skip', weights_skip)
            if self.use_biases:
                tf.histogram_summary(layer + '_biases_filter', filter_bias)
                tf.histogram_summary(layer + '_biases_gate', gate_bias)
                tf.histogram_summary(layer + '_biases_dense', dense_bias)

        return skip_contribution, input + transformed

    def _create_network(self, input_batch):
        with tf.name_scope('layer_stack'):
            # Put 32 NULL entries on either side of the input time
            # series (dimension 1), and 0 padding before and after on dim 0.
            input_batch = tf.pad(input_batch, [[0, 0], [32, 32]])

            current_layer = input_batch
            for i in range(LAYER_COUNT):
                with tf.name_scope('layer{}'.format(i)):
                    output, current_layer = self._create_layer(
                        current_layer, i)
                    skip_outs.append(output)

        with tf.name_scope('postprocessing'):
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']
            b1 = self.variables['postprocessing']['bias1']
            b2 = self.variables['postprocessing']['bias2']

            total = sum(skip_outs + b1)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            transformed2 = tf.nn.relu(conv1 + b2)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")

        with tf.name_scope('upsampling'):
            # Reshape so we can use 2d conv on it: height = 1
            input_shape = tf.shape(conv2)
            conv2 = tf.reshape(input_shape[0], 1, input_shape[1],
                               input_shape[2])

            upsampled_shape = [input_shape[0], 1, input_shape[1]*UPSAMPLE_RATE,
                               input_shape[2]]
            filt = self.variables['upsampling']['filter']
            # 2d transpose conv with height = 1, width = characters.
            upsampled = tf.nn.conv2d_transpose(upsampled,
                                               filt,
                                               upsampled_shape,
                                               [1, UPSAMPLE_RATE],
                                               padding='VALID',
                                               name='upsampled')
            # Remove the dummy "height=1" dimension we added for the 2d conv,
            # to bring it back to batch x duration x channels.
            upsampled = tf.reshape(upsampled, [input_shape[0],
                                               -1,
                                               self.local_condition_channels])
        return upsampled

    def upsample(self, input_text):
        with tf.name_scope('text_conv_net'):
            upsampled = self._create_net(input_text)
        return upsampled
