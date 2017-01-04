import numpy as np
import tensorflow as tf
from .ops import quantize_sample_density

DENSITY_QUANT_LEVELS = 50
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
        initializer = tf.truncated_normal(shape, mean=0.0, stddev=0.3,
                                dtype=tf.float32)
        variable = tf.Variable(initializer, name=name)
        return variable

def create_bias_variable(name, shape, value=0.0):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)

class ConvNetModel(object):
    def __init__(self,
                 encoder_channels,
                 histograms,
                 output_channels,
                 local_condition_channels,
                 layer_count=None,
                 dilations=None,
                 gated_linear=False,
                 density_conditioned=False):
        self.encoder_channels = encoder_channels
        self.histograms = histograms
        self.output_channels = output_channels
        self.local_condition_channels = local_condition_channels
        if layer_count is None:
            self.layer_count = len(dilations)
        elif dilations is not None:
            assert(len(dilations) == layer_count)
            self.layer_count=layer_count
        else:
            assert(False)

        print("self.layer_count:{}".format(self.layer_count))
        self.skip_cuts = []
        self.output_shapes = []
        self.output_width = None
        self.dilations = dilations
        self.text_embedded_shape = None
        self.layer_out_shapes = []
        self.embedding_shape = None
        self.text_shape = None
        self.gated_linear = gated_linear
        # True if this net is conditioning on density.
        self.density_conditioned = density_conditioned
        self.sample_density = None

        self.variables = self._create_variables()

    def _receptive_field(self):
        if self.dilations is None:
            # Single-strided conv filter witdth 2.
            return 2 + self.layer_count - 1
        else:
            # Dilated conv.
            max_dilation = max(self.dilations)
            num_stacks = self.dilations.count(max_dilation)
            size = max_dilation*2 + (num_stacks-1)*(max_dilation*2-1)
            return size

    def _create_variables(self):
        var = dict()
        with tf.variable_scope('encoder_convnet'):
            with tf.variable_scope('embeddings'):
                layer = dict()
                layer['text_embedding'] = create_embedding_table(
                    'text_embedding',
                    [CHARACTER_CARDINALITY, self.encoder_channels])
                if self.density_conditioned:
                    layer['density_embedding'] = create_embedding_table(
                        'density_embedding',
                        [DENSITY_QUANT_LEVELS, self.encoder_channels])
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

                        if self.density_conditioned:
                            current['sd_filt'] = create_variable(
                                'sd_filt',
                                [1, self.encoder_channels,
                                 self.encoder_channels])
                            current['sd_gate'] = create_variable(
                                'sd_gate',
                                [1, self.encoder_channels,
                                 self.encoder_channels])
                            current['sd_filt_bias'] = create_bias_variable(
                                'sd_filt_bias',
                                [self.encoder_channels])
                            current['sd_gate_bias'] = create_bias_variable(
                                'sd_gate_bias',
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
                    'bias1', [self.output_channels], value=0.0)
                current['bias2'] = create_bias_variable(
                    'bias2', [self.output_channels], value=0.0)
                current['projection_filter'] = create_variable(
                    'lc_proj_filter',
                    [1,
                     self.output_channels,
                     self.local_condition_channels])

                var['postprocessing'] = current

            with tf.variable_scope('upsampling'):
                # filter for tf.nn.conv2d_transpose, with height = 1 to achieve
                # a 1d deconv.
                current = dict()
#                current['filter'] = create_variable(
#                    'upsample_filter',
#                    [1,
#                     self.upsample_rate,
#                     self.local_condition_channels,
#                     self.output_channels])


                var['upsampling'] = current
        return var

    def _create_layer(self, input, layer_index, output_width, dilation,
                      density_embedding):

        is_last_layer = (layer_index == (self.layer_count - 1))
        variables = self.variables['layer_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']
        filter_bias = variables['filter_bias']
        gate_bias = variables['gate_bias']


        dilation_rate = [dilation] if dilation is not None else None

        conv_filter = tf.nn.convolution(input=input,
                                        filter=weights_filter,
                                        padding='VALID',
                                        dilation_rate=dilation_rate,
                                        name='filter',
                                        data_format='NWC') + filter_bias
        conv_gate = tf.nn.convolution(input=input,
                                      filter=weights_gate,
                                      padding='VALID',
                                      dilation_rate=dilation_rate,
                                      name='gate',
                                      data_format='NWC') + gate_bias

        if self.density_conditioned:
            assert(density_embedding is not None)
            density_filt_weights = variables['sd_filt']
            density_filt_bias = variables['sd_filt_bias']
            density_gate_weights = variables['sd_gate']
            density_gate_bias = variables['sd_gate_bias']

            conv_filter += tf.nn.conv1d(value=density_embedding,
                                        filters=density_filt_weights,
                                        stride=1,
                                        padding="SAME",
                                        name='sd_filter') + \
                         density_filt_bias

            conv_gate += tf.nn.conv1d(value=density_embedding,
                                      filters=density_gate_weights,
                                      stride=1,
                                      padding="SAME",
                                      name='sd_gates') + \
                           density_gate_bias
        else:
            assert(density_embedding is None)


        if self.gated_linear:
            out = conv_filter * tf.sigmoid(conv_gate)
        else:
            out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        self.output_shapes.append(tf.shape(out))
        skip_cut = (tf.shape(out)[1] - output_width) // 2

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

    def _density_embedding(self):
        if not self.density_conditioned:
            return None
        quantized_density = quantize_sample_density(
            self.sample_density, DENSITY_QUANT_LEVELS)
        table = self.variables['embeddings']['density_embedding']
        density_embedding = tf.nn.embedding_lookup(table,
                                                   quantized_density)
        shape = [1, 1, self.encoder_channels]
        density_embedding = tf.reshape(density_embedding, shape)

        return density_embedding

    def _create_network(self, input_batch, audio_length):
        with tf.name_scope('layer_stack'):
            skip_outs = []
            current_layer = input_batch
            density_embedding = self._density_embedding()
            for i in range(self.layer_count):
                dilation = self.dilations[i] if self.dilations is not None \
                    else None
                with tf.name_scope('layer{}'.format(i)):
                    output, current_layer = self._create_layer(
                        current_layer, i, audio_length, dilation,
                        density_embedding)
                    skip_outs.append(output)
                    if current_layer is not None:
                        self.layer_out_shapes.append(tf.shape(current_layer))

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

            filt = self.variables['postprocessing']['projection_filter']
            conv_out = tf.nn.conv1d(conv2, filt, stride=1,
                                       padding="SAME", name="lc_projection")

#            upsampled_shape = [1,
#                               1,
#                               input_shape[2] * self.upsample_rate,
#                               self.local_condition_channels]
#            filt = self.variables['upsampling']['filter']
#            # 2d transpose conv with height = 1, width = characters.
#            upsampled = tf.nn.conv2d_transpose(conv2,
#                                               filt,
#                                               upsampled_shape,
#                                               [1, 1, self.upsample_rate, 1],
#                                               padding='VALID',
#                                               name='upsampled')
        return conv_out

    def _upsample(self, embedding, audio_length):
        with tf.name_scope('upsampling'):

            # Number of samples per character.
            self.sample_density = tf.cast(audio_length, dtype=tf.float32) /  \
                tf.cast(self.text_shape[0], dtype=tf.float32)

            # Number of samples we've currently got if we preserve density.
            number_samples = self.sample_density * tf.cast(
                tf.shape(embedding)[1], dtype=tf.float32)
            number_samples = tf.cast(tf.ceil(number_samples), dtype=tf.int32)

            # Reshape so we can use image resize on it: height = 1
            embedding = tf.expand_dims(embedding, axis=1)
            upsampled = tf.image.resize_bilinear(embedding,
                                                 [1, number_samples])
            # Remove the dummy "height=1" dimension we added for the resize
            # to bring it back to batch x duration x channels.
            upsampled = tf.squeeze(upsampled, axis=1)

            # Desired number of samples
            audio_padding = self._receptive_field() // 2
            desired_samples = audio_length + 2 * audio_padding
            cut_size = (number_samples - desired_samples) // 2

            upsampled = tf.slice(upsampled,
                                 [0, cut_size, 0],
                                 [-1, desired_samples, -1])
            return upsampled


    def _compute_character_padding(self, audio_pad_size, audio_length):
        float_padding =  tf.cast(audio_pad_size * self.text_shape[0],
                                 dtype=tf.float32) / tf.cast(audio_length,
                                                             dtype=tf.float32)
        return tf.cast(tf.ceil(float_padding), dtype=tf.int32)

    def _embed_ascii(self, ascii, audio_length):
        with tf.name_scope('input_embedding'):
            audio_pad_size = self._receptive_field() // 2
            char_pad = self._compute_character_padding(audio_pad_size,
                audio_length)
            # Put entries of 0 on either side of the input time
            # series (dimension 1).
            padded_ascii = tf.pad(ascii, [[char_pad, char_pad]]) # dim1
            embedding_table = self.variables['embeddings']['text_embedding']
            embedding = tf.nn.embedding_lookup(embedding_table,
                                               padded_ascii)
            self.embedding_shape = tf.shape(embedding)
            shape = [1, -1, self.encoder_channels]
            embedding = tf.reshape(embedding, shape)
        return embedding

    def upsample(self, input_text, audio_length):
        with tf.name_scope('text_conv_net'):
            self.text_shape = tf.shape(input_text)
            embedding = self._embed_ascii(input_text, audio_length)
            upsampled = self._upsample(embedding, audio_length)
            out = self._create_network(upsampled, audio_length)
        return out
