import numpy as np
import tensorflow as tf

CHARACTER_CARDINALITY = 256
FILTER_WIDTH = 2

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
                 local_condition_channels,
                 upsample_rate,
                 layer_count):
        self.batch_size = batch_size
        self.encoder_channels = encoder_channels
        self.histograms = histograms
        self.output_channels = output_channels
        self.local_condition_channels = local_condition_channels
        self.upsample_rate = upsample_rate
        self.layer_count = layer_count
        self.skip_cuts = []
        self.output_shapes = []
        self.output_width = None

        self.variables = self._create_variables()

        assert self.batch_size == 1

    def _receptive_field(self):
        # return 2 ** self.layer_count
        return 2 + self.layer_count - 1

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
                for i in range(self.layer_count):
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
                        if i != self.layer_count - 1:
                            current['dense'] = create_variable(
                                'dense',
                                [1,
                                 self.encoder_channels,
                                 self.encoder_channels])
                            current['dense_bias'] = create_bias_variable(
                                'dense_bias',
                                [self.encoder_channels])
                        var['layer_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess1'] = create_variable(
                    'postprocess1',
                    [1, self.output_channels, self.output_channels])
                current['postprocess2'] = create_variable(
                    'postprocess2',
                    [1, self.output_channels, self.output_channels])
                current['bias1'] = create_bias_variable(
                    'bias1', [self.output_channels], value=-0.2)
                current['bias2'] = create_bias_variable(
                    'bias2', [self.output_channels], value=-0.2)
                var['postprocessing'] = current

            with tf.variable_scope('upsampling'):
                # filter for tf.nn.conv2d_transpose, with height = 1 to achieve
                # a 1d deconv.
                current = dict()
                current['filter'] = create_variable(
                    'upsample_filter',
                    [1,
                     self.upsample_rate,
                     self.local_condition_channels,
                     self.output_channels])
                var['upsampling'] = current
        return var

    def _create_layer(self, input, layer_index, output_width):
        is_last_layer = (layer_index == (self.layer_count - 1))
        variables = self.variables['layer_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']
        filter_bias = variables['filter_bias']
        gate_bias = variables['gate_bias']

        conv_filter = tf.nn.conv1d(input,
                                   weights_filter,
                                   stride=1, # 2,
                                   padding="VALID",
                                   name="filter") + filter_bias

        conv_gate = tf.nn.conv1d(input,
                                 weights_gate,
                                 stride=1, # 2,
                                 padding="VALID",
                                 name="gate") + gate_bias

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        self.output_shapes.append(tf.shape(out))
        skip_cut = (tf.shape(out)[1] - output_width) / 2

        self.skip_cuts.append(skip_cut)

        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, output_width, -1])

        if not is_last_layer:
            # Last residual layer output does not connect to anything.
            # The 1x1 conv to produce the residual output
            weights_dense = variables['dense']
            dense_bias = variables['dense_bias']
            transformed = tf.nn.conv1d(out, weights_dense, stride=1,
                                       padding="SAME", name="dense")
            transformed += dense_bias

        # The 1x1 conv to produce the skip output
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
                out_skip, weights_skip, stride=1, padding="SAME", name="skip")

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

        if not is_last_layer:
            input = tf.slice(input,
                             [0, 0, 0],
                             [-1, tf.shape(transformed)[1], -1])
            return skip_contribution, input + transformed
        else:
            return skip_contribution, None

    def _create_network(self, input_batch):
        output_width = tf.shape(input_batch)[1] - self._receptive_field() + 1
        self.output_width = output_width

        with tf.name_scope('layer_stack'):
            skip_outs = []
            current_layer = input_batch
            for i in range(self.layer_count):
                with tf.name_scope('layer{}'.format(i)):
                    output, current_layer = self._create_layer(
                        current_layer, i, output_width)
                    skip_outs.append(output)

        with tf.name_scope('postprocessing'):
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']
            b1 = self.variables['postprocessing']['bias1']
            b2 = self.variables['postprocessing']['bias2']

            total = sum(skip_outs) + b1
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            transformed2 = tf.nn.relu(conv1 + b2)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")

        with tf.name_scope('upsampling'):
            # Reshape so we can use 2d conv on it: height = 1
            # input_shape = tf.shape(conv2)
            # conv2 = tf.reshape(conv2, [input_shape[0], 1, input_shape[1],
            #                         input_shape[2]])
            conv2 = tf.expand_dims(conv2, axis=1)

            input_shape = tf.shape(conv2)
            upsampled_shape = [1,
                               1,
                               input_shape[2] * self.upsample_rate,
                               self.local_condition_channels]
            filt = self.variables['upsampling']['filter']
            # 2d transpose conv with height = 1, width = characters.
            upsampled = tf.nn.conv2d_transpose(conv2,
                                               filt,
                                               upsampled_shape,
                                               [1, 1, self.upsample_rate, 1],
                                               padding='VALID',
                                               name='upsampled')
            # Remove the dummy "height=1" dimension we added for the 2d conv,
            # to bring it back to batch x duration x channels.
#            upsampled = tf.reshape(upsampled, [input_shape[0],
#                                               -1,
#                                               self.local_condition_channels])
            upsampled = tf.squeeze(upsampled, axis=1)
        return upsampled

    def _embed_ascii(self, ascii):
        with tf.name_scope('input_embedding'):

            # Put 32 NULL entries on either side of the input time
            # series (dimension 1), and 0 padding before and after on dim 0.
            pad_size = self._receptive_field() / 2
            input_batch = tf.pad(ascii, [[0, 0], # pad for dim 0
                                         [pad_size, pad_size]]) # dim1

            embedding_table = self.variables['embeddings']['text_embedding']
            embedding = tf.nn.embedding_lookup(embedding_table,
                                               ascii)
            shape = [self.batch_size, -1, self.encoder_channels]
            embedding = tf.reshape(embedding, shape)
        return embedding

    def upsample(self, input_text):
        with tf.name_scope('text_conv_net'):
            embedding = self._embed_ascii(input_text)
            upsampled = self._create_network(embedding)
        return upsampled
