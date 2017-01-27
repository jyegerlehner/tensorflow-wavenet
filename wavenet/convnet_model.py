import numpy as np
import tensorflow as tf
from .ops import (quantize_sample_density, create_variable,
                  create_embedding_table, create_bias_variable, show_params)
from .paramspec import (create_vars, StoredParm, ComputedParm,
                        ParamTree, create_var)

DENSITY_QUANT_LEVELS = 50
CHARACTER_CARDINALITY = 256
FILTER_WIDTH = 2
TOP_NAME = 'encoder_convnet'

class ConvNetModel(object):
    def __init__(self,
                 encoder_channels,
                 histograms,
                 output_channels,
                 local_condition_channels,
                 layer_count=None,
                 dilations=None,
                 gated_linear=False,
                 density_conditioned=False,
                 compute_the_params=False,
                 non_computed_params=None):
        self.encoder_channels = encoder_channels
        self.histograms = histograms
        self.output_channels = output_channels
        self.local_condition_channels = local_condition_channels
        self.layer_count = layer_count
        if layer_count is None:
            self.layer_count = len(dilations)
        elif dilations is not None:
            assert(len(dilations) == layer_count)

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
        self.compute_the_params = compute_the_params
        self.non_computed_params = non_computed_params
        self.param_specs=None
        self.variables = self._create_vars()

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

    def _make_spec(self, name, shape, kind):
        def has_match(name, tokens):
            match = False
            for token in tokens:
                if token in name:
                    match = True
                    break
            return match

        if self.compute_the_params:
            # non_computed_params is list of tokens, such that if that token
            # appears in the parameter name, it is not computed.
            if has_match(name, self.non_computed_params):
                return StoredParm(name=name, shape=shape, kind=kind)
            else:
                return ComputedParm(name=name, shape=shape, kind=kind)
        else:
            return StoredParm(name=name, shape=shape, kind=kind)


    def _add_recursively(self, name, newvars, vars):
        if isinstance(newvars, dict):
            for child_name in newvars.keys():
                if isinstance(newvars[child_name], dict):
                    assert child_name in vars
                    # Both are dicts
                    self._add_recursively(child_name, newvars[child_name],
                                          vars[child_name])
                else:
                    # newvars[name] is a param tensor.
                    self._add_recursively(child_name, newvars[child_name],
                                          vars)
        else:
            # newvars is a variable produced by param_producer_model.
            vars[name] = newvars


    # Given a recursive dict of parameters produced by a param_producer_model,
    # add them in to the vars dictionary. Assumes the non-computed parameters
    # have already been created and added to the self.variables, along with the
    # levels in the tree.
    def merge_params(self, newvars):
        self._add_recursively('', newvars, self.variables)


    def create_param_specs(self):
        t = ParamTree(TOP_NAME)
        c = t.add_child('embeddings')
        c.add_param(self._make_spec(name='text_embedding',
                                    shape=[CHARACTER_CARDINALITY,
                                        self.encoder_channels],
                                    kind='embedding'))
        if self.density_conditioned:
            c.add_param(self._make_spec(name='density_embedding',
                                   shape=[DENSITY_QUANT_LEVELS,
                                          self.encoder_channels],
                                   kind='embedding'))

        c = t.add_child('layer_stack')
        for layer in range(self.layer_count):
            l = c.add_child('layer{}'.format(layer))
            l.add_param(self._make_spec(name='filter',
                                    shape=[FILTER_WIDTH,
                                           self.encoder_channels,
                                           self.encoder_channels],
                                    kind='filter'))
            l.add_param(self._make_spec(name='gate',
                                   shape=[FILTER_WIDTH,
                                          self.encoder_channels,
                                          self.encoder_channels],
                                   kind='filter'))
            l.add_param(self._make_spec(name='skip',
                                   shape=[1,
                                          self.encoder_channels,
                                          self.output_channels],
                                   kind='filter'))
            l.add_param(self._make_spec(name='filter_bias',
                                   shape=[self.encoder_channels],
                                   kind='bias'))
            l.add_param(self._make_spec(name='gate_bias',
                                   shape=[self.encoder_channels],
                                   kind='bias'))
            if self.density_conditioned:
                l.add_param(self._make_spec(name='sd_filt',
                                       shape=[1,
                                              self.encoder_channels,
                                              self.encoder_channels],
                                       kind='filter'))
                l.add_param(self._make_spec(name='sd_gate',
                                       shape=[1,
                                              self.encoder_channels,
                                              self.encoder_channels],
                                       kind='filter'))
                l.add_param(self._make_spec(name='sd_filt_bias',
                                       shape=[self.encoder_channels],
                                       kind='bias'))
                l.add_param(self._make_spec(name='sd_gate_bias',
                                       shape=[self.encoder_channels],
                                       kind='bias'))
            if layer != self.layer_count-1:
                l.add_param(self._make_spec(name='dense',
                                       shape=[1,
                                              self.encoder_channels,
                                              self.encoder_channels],
                                       kind='filter'))
                l.add_param(self._make_spec(name='dense_bias',
                                       shape=[self.encoder_channels],
                                       kind='bias'))

        c = t.add_child('postprocessing')
        c.add_param(self._make_spec(name='postprocess1',
                               shape=[1,
                                      self.output_channels,
                                      self.output_channels],
                               kind='filter'))
        c.add_param(self._make_spec(name='postprocess2',
                               shape=[1,
                                      self.output_channels,
                                      self.output_channels],
                               kind='filter'))
        c.add_param(self._make_spec(name='bias1',
                               shape=[self.output_channels],
                               kind='bias'))
        c.add_param(self._make_spec(name='bias2',
                               shape=[self.output_channels],
                               kind='bias'))
        c.add_param(self._make_spec(name='lc_proj_filter',
                               shape=[1,
                                      self.output_channels,
                                      self.local_condition_channels],
                               kind='filter'))
        return t

    def param_dict():
        return self.variables

    def _create_vars(self):
        self.param_specs = self.create_param_specs()

        return create_vars(spec_tree=self.param_specs,
                                  # False because these are the stored, not
                                  # computed params.
                                  computed_not_stored=False,
                                  parm_factory=create_var)


    def _create_layer(self, input, layer_index, output_width, dilation,
                      density_embedding):

        is_last_layer = (layer_index == (self.layer_count - 1))
        variables = self.variables[TOP_NAME]['layer_stack'][
                        'layer{}'.format(layer_index)]

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

    def _density_embedding(self, sample_density):
        if not self.density_conditioned:
            return None
        quantized_density = quantize_sample_density(
            sample_density, DENSITY_QUANT_LEVELS)
        table = self.variables[TOP_NAME]['embeddings']['density_embedding']
        density_embedding = tf.nn.embedding_lookup(table,
                                                   quantized_density)
        shape = [1, 1, self.encoder_channels]
        density_embedding = tf.reshape(density_embedding, shape)

        return density_embedding

    def _create_network(self, input_batch, ascii_length, sample_density):
        with tf.name_scope('layer_stack'):
            skip_outs = []
            current_layer = input_batch
            density_embedding = self._density_embedding(sample_density)
            for i in range(self.layer_count):
                dilation = self.dilations[i] if self.dilations is not None \
                    else None
                with tf.name_scope('layer{}'.format(i)):
                    output, current_layer = self._create_layer(
                        current_layer, i, ascii_length, dilation,
                        density_embedding)
                    skip_outs.append(output)
                    if current_layer is not None:
                        self.layer_out_shapes.append(tf.shape(current_layer))

        with tf.name_scope('postprocessing'):
            postproc_vars = self.variables[TOP_NAME]['postprocessing']
            w1 = postproc_vars['postprocess1']
            w2 = postproc_vars['postprocess2']
            b1 = postproc_vars['bias1']
            b2 = postproc_vars['bias2']

            total = sum(skip_outs) + b1
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            transformed2 = tf.nn.relu(conv1 + b2)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")

            filt = postproc_vars['lc_proj_filter']
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

    def _upsample(self, embedding, audio_length, sample_density):
        with tf.name_scope('upsampling'):
            # Number of samples we've currently got if we preserve density.
            number_samples = sample_density * tf.cast(
                tf.shape(embedding)[1], dtype=tf.float32)
            number_samples = tf.cast(tf.ceil(number_samples), dtype=tf.int32)

            with tf.control_dependencies([tf.assert_equal(number_samples,
                                          audio_length)]):
                # Reshape so we can use image resize on it: height = 1
                embedding = tf.expand_dims(embedding, axis=1)
#            upsampled = tf.image.resize_bilinear(embedding,
#                                                 [1, number_samples])
                upsampled = tf.image.resize_bilinear(embedding,
                                                    [1, number_samples])
                # Remove the dummy "height=1" dimension we added for the resize
                # to bring it back to batch x duration x channels.
                upsampled = tf.squeeze(upsampled, axis=1)

#            # Desired number of samples
#            audio_padding = self._receptive_field() // 2
#            desired_samples = audio_length + 2 * audio_padding
#            cut_size = (number_samples - desired_samples) // 2

#            upsampled = tf.slice(upsampled,
#                                 [0, cut_size, 0],
#                                 [-1, desired_samples, -1])
            return upsampled


    def _compute_character_padding(self, audio_pad_size, audio_length):
        float_padding =  tf.cast(audio_pad_size * self.text_shape[0],
                                 dtype=tf.float32) / tf.cast(audio_length,
                                                             dtype=tf.float32)
        return tf.cast(tf.ceil(float_padding), dtype=tf.int32)

    def _embed_ascii(self, ascii):
        with tf.name_scope('input_embedding'):
            char_pad = self._receptive_field() // 2
            # Put entries of 0 on either side of the input time
            # series (dimension 1). This will make the output of the
            # character-resolution portion of the net have exactly the same
            # width as the input string (width=number of characters).
            padded_ascii = tf.pad(ascii, [[char_pad, char_pad]]) # dim1
            embedding_table = self.variables[TOP_NAME]['embeddings'
                                  ]['text_embedding']
            embedding = tf.nn.embedding_lookup(embedding_table,
                                               padded_ascii)
            self.embedding_shape = tf.shape(embedding)
            shape = [1, -1, self.encoder_channels]
            embedding = tf.reshape(embedding, shape)
        return embedding

    def upsample(self, input_text, audio_length, sample_density):
        with tf.name_scope('text_conv_net'):
            self.text_shape = tf.shape(input_text)
            embedding = self._embed_ascii(input_text)
            # The 'output' is still character-resolution. Has the same width
            # as the input ascii string.
            output = self._create_network(embedding, self.text_shape[0],
                                                      sample_density)
            upsampled = self._upsample(output, audio_length, sample_density)

        return upsampled

