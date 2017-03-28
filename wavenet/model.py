import numpy as np
import tensorflow as tf

from .ops import (causal_conv, mu_law_encode, mu_law_decode, create_variable,
                  create_embedding_table, create_bias_variable)
from .paramspec import (create_vars, StoredParm, ComputedParm,
                        ParamTree, create_var)
from .frequency_domain_loss import (output_to_probs, FrequencyDomainLoss,
                                    probs_to_entropy_bits)


TOP_NAME = 'wavenet'


class WaveNetModel(object):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel( dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    '''

    def __init__(self,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 elu_not_relu,
                 quantization_channels=2**8,
                 use_biases=False,
                 scalar_input=False,
                 initial_filter_width=32,
                 histograms=False,
                 global_condition_channels=None,
                 global_condition_cardinality=None,
                 local_condition_channels=None,
                 gated_linear=False,
                 compute_the_params=False,
                 non_computed_params=None,
                 frequency_domain_loss=False,
                 sample_rate=None):
        '''Initializes the WaveNet model.

        Args:
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            quantization_channels: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            scalar_input: Whether to use the quantized waveform directly as
                input to the network instead of one-hot encoding it.
                Default: False.
            initial_filter_width: The width of the initial filter of the
                convolution applied to the scalar input. This is only relevant
                if scalar_input=True.
            histograms: Whether to store histograms in the summary.
                Default: False.
            global_condition_channels: Number of channels in (embedding
                size) of global conditioning vector. None indicates there is
                no global conditioning.
            global_condition_cardinality: Number of mutually exclusive
                categories to be embedded in global condition embedding. If
                not None, then this implies that global_condition tensor
                specifies an integer selecting which of the N global condition
                categories, where N = global_condition_cardinality. If None,
                then the global_condition tensor is regarded as a vector which
                must have dimension global_condition_channels.
            gated_linear: True iff we use linear instead of hypertan in the
                gating units.
            compute_the_params: True iff we compute params in a hypernet
                arrangement.
            non_computed_params: List of tokens indicating which parameters
                are not computed. Only has an effect if compute_the_params
                is True.
            frequency_domain_loss: True iff loss should be computed in
                frequency domain.

        '''
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.histograms = histograms
        self.global_condition_channels = global_condition_channels
        self.global_condition_cardinality = global_condition_cardinality
        self.local_condition_channels = local_condition_channels
        self.elu_not_relu = elu_not_relu
        # Add one category for "blank"
        self.softmax_channels = self.quantization_channels
        self.gated_linear = gated_linear
        self.compute_the_params = compute_the_params
        self.non_computed_params = non_computed_params
        self.indices = None
        self.labs = None
        self.raw_output_shape = None
        self.lc_shape = None
        self.disc_input_shape = None
        self.extended_shape = None
        if frequency_domain_loss:
            # Lowest audible freuency
            LOWEST_FREQUENCY_OF_LOSS = 20
            LOW_FREQ_PERIOD = sample_rate // LOWEST_FREQUENCY_OF_LOSS
            self.freqloss = FrequencyDomainLoss(
                max_period=LOW_FREQ_PERIOD,
                quantization_levels=quantization_channels)
        else:
            self.freqloss = None

        (self.variables, self.orthogonal_reg_losses) = self._create_vars()
        self.entropy_loss = None
        # Set this to false if you want only frequency domain loss.
        self.cross_entropy_loss = True

    def _make_spec(self, name, shape, kind, initial_value=None,
                   regularization=False):
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
                return StoredParm(
                    name=name, shape=shape, kind=kind,
                    initial_value=initial_value,
                    regularization=regularization)
            else:
                return ComputedParm(
                    name=name, shape=shape, kind=kind,
                    initial_value=initial_value,
                    regularization=regularization)
        else:
            return StoredParm(
                name=name, shape=shape, kind=kind,
                initial_value=initial_value,
                regularization=regularization)

    def _create_vars(self):
        self.param_specs = self.create_param_specs()

        self.param_specs = self.param_specs  # self.param_specs.children[0]
        return create_vars(spec_tree=self.param_specs,
                           computed_not_stored=False,
                           parm_factory=create_var)

    def create_param_specs(self):
        '''This function creates all specs of all the parameters in the model.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        t = ParamTree(TOP_NAME)
        c = t.add_child('embeddings')
        # We only look up the embedding if we are conditioning on a
        # set of mutually-exclusive categories. We can also condition
        # on an already-embedded dense vector, in which case it's
        # given to us and we don't need to do the embedding lookup.
        # Still another alternative is no global condition at all, in
        # which case we also don't do a tf.nn.embedding_lookup.
        if self.global_condition_cardinality is not None:
            c.add_param(self._make_spec(
                name='gc_embedding',
                shape=[self.global_condition_cardinality,
                       self.global_condition_channels],
                kind='embedding',
                regularization=True)) #

        c.add_param(self._make_spec(
            name='input_embedding',
            shape=[self.softmax_channels,
                   self.residual_channels],
            kind='embedding',
            regularization=True)) #

        c = t.add_child('dilated_stack')
        for i, dilation in enumerate(self.dilations):
            # with tf.variable_scope('layer{}'.format(i)):
            l = c.add_child('layer{}'.format(i))
            if self.local_condition_channels is not None:
                input_channels = self.residual_channels + \
                                 self.local_condition_channels
            else:
                input_channels = self.residual_channels

            l.add_param(self._make_spec(
                name='filter',
                shape=[self.filter_width,
                       input_channels,
                       self.dilation_channels],
                kind='filter',
                regularization=True)) #
            l.add_param(self._make_spec(
                name='gate',
                shape=[self.filter_width,
                       input_channels,
                       self.dilation_channels],
                kind='filter',
                regularization=True)) #
            l.add_param(self._make_spec(
                name='dense',
                shape=[1,
                       self.dilation_channels,
                       self.residual_channels],
                kind='filter',
                regularization=True))
            l.add_param(self._make_spec(
                name='skip',
                shape=[1,
                       self.dilation_channels,
                       self.skip_channels],
                kind='filter',
                regularization=True))

            if self.global_condition_channels is not None:
                l.add_param(self._make_spec(
                    name='gc_gateweights',
                    shape=[1, self.global_condition_channels,
                           self.dilation_channels],
                    kind='filter',
                    regularization=True))

                l.add_param(self._make_spec(
                    name='gc_filtweights',
                    shape=[1, self.global_condition_channels,
                           self.dilation_channels],
                    kind='filter',
                    regularization=True))

            if self.use_biases:
                l.add_param(self._make_spec(
                    name='filter_bias',
                    shape=[self.dilation_channels],
                    kind='bias',
                    regularization=True))
                l.add_param(self._make_spec(
                    name='gate_bias',
                    shape=[self.dilation_channels],
                    kind='bias',
                    regularization=True))
                l.add_param(self._make_spec(
                    name='dense_bias',
                    shape=[self.residual_channels],
                    kind='bias',
                    regularization=True))

        c = t.add_child('postprocessing')
        c.add_param(self._make_spec(
            name='postprocess1',
            shape=[1, self.skip_channels, self.skip_channels],
            kind='filter',
            regularization=True))

        c.add_param(self._make_spec(
            name='postprocess2',
            shape=[1, self.skip_channels, self.softmax_channels],
            kind='filter',
            regularization=True))

        if self.use_biases:
            c.add_param(self._make_spec(
                name='postprocess1_bias',
                shape=[self.skip_channels],
                kind='bias',
                regularization=True))
            c.add_param(self._make_spec(
                name='postprocess2_bias',
                shape=[self.skip_channels],
                kind='bias',
                regularization=True))

#        if self.freqloss is not None:
#            c.add_param(self._make_spec(
#                name='temperature',
#                shape=[],
#                kind='filter',
#                initial_value=2.0))

        return t

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

    def _create_dilation_layer(self, input_batch, layer_index, dilation,
                               global_condition_batch, local_condition_batch,
                               is_last_layer):
        '''Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
             global_conditioning_batch: Tensor containing the global data upon
                 which the output is to be conditioned upon. Shape:
                 [batch size, 1, channels]. The 1 is for the axis
                 corresponding to time so that the result is broadcast to
                 all time steps.

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and global conditioning
        are omitted due to the limits of ASCII art.

        '''
        variables = self.variables[TOP_NAME]['dilated_stack'][
                      'layer{}'.format(layer_index)]

        weights_filter = variables['filter']
        weights_gate = variables['gate']

        if local_condition_batch is not None:
            conv_filt_inp = tf.concat(axis=2,
                                      values=[local_condition_batch,
                                              input_batch])
        else:
            conv_filt_inp = input_batch

        conv_filter = causal_conv(conv_filt_inp, weights_filter, dilation)
        conv_gate = causal_conv(conv_filt_inp, weights_gate, dilation)

        if global_condition_batch is not None:
            weights_gc_filter = variables['gc_filtweights']
            conv_filter = conv_filter + tf.nn.conv1d(global_condition_batch,
                                                     weights_gc_filter,
                                                     stride=1,
                                                     padding="SAME",
                                                     name="gc_filter")
            weights_gc_gate = variables['gc_gateweights']
            conv_gate = conv_gate + tf.nn.conv1d(global_condition_batch,
                                                 weights_gc_gate,
                                                 stride=1,
                                                 padding="SAME",
                                                 name="gc_gate")

        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)


        if self.gated_linear:
            out = conv_filter * tf.sigmoid(conv_gate)
        else:
            out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        if not is_last_layer:
            # Last residual layer output does not connect to anything.

            # The 1x1 conv to produce the residual output
            weights_dense = variables['dense']
            transformed = tf.nn.conv1d(
                out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            if not is_last_layer:
                dense_bias = variables['dense_bias']
                transformed = transformed + dense_bias
            skip_contribution = skip_contribution

        if self.histograms:
            layer = 'layer{}'.format(layer_index)
            tf.histogram_summary(layer + '_filter', weights_filter)
            tf.histogram_summary(layer + '_gate', weights_gate)
            if not is_last_layer:
                tf.histogram_summary(layer + '_dense', weights_dense)
            tf.histogram_summary(layer + '_skip', weights_skip)
            if self.use_biases:
                tf.histogram_summary(layer + '_biases_filter', filter_bias)
                tf.histogram_summary(layer + '_biases_gate', gate_bias)
                tf.histogram_summary(layer + '_biases_dense', dense_bias)

        if is_last_layer:
            return skip_contribution, None
        else:
            return skip_contribution, input_batch + transformed

    def _generator_conv(self, input_batch, state_batch, weights):
        '''Perform convolution for a single convolutional processing step.'''
        # TODO generalize to filter_width > 2
        past_weights = weights[0, :, :]
        curr_weights = weights[1, :, :]
        output = tf.matmul(state_batch, past_weights) + tf.matmul(
            input_batch, curr_weights)
        return output

#    def _generator_causal_layer(self, input_batch, state_batch):
#        with tf.name_scope('causal_layer'):
#            weights_filter = self.variables['causal_layer']['filter']
#            output = self._generator_conv(
#                input_batch, state_batch, weights_filter)
#        return output

    def _generator_dilation_layer(self, input_batch, state_batch, layer_index,
                                  dilation, global_condition_batch):
        variables = self.variables[TOP_NAME]['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']
        output_filter = self._generator_conv(
            input_batch, state_batch, weights_filter)
        output_gate = self._generator_conv(
            input_batch, state_batch, weights_gate)

        if global_condition_batch is not None:
            global_condition_batch = tf.reshape(global_condition_batch,
                                                shape=(1, -1))
            weights_gc_filter = variables['gc_filtweights']
            weights_gc_filter = weights_gc_filter[0, :, :]
            output_filter += tf.matmul(global_condition_batch,
                                       weights_gc_filter)
            weights_gc_gate = variables['gc_gateweights']
            weights_gc_gate = weights_gc_gate[0, :, :]
            output_gate += tf.matmul(global_condition_batch,
                                     weights_gc_gate)

        if self.use_biases:
            output_filter = output_filter + variables['filter_bias']
            output_gate = output_gate + variables['gate_bias']

        out = tf.tanh(output_filter) * tf.sigmoid(output_gate)

        weights_dense = variables['dense']
        transformed = tf.matmul(out, weights_dense[0, :, :])
        if self.use_biases:
            transformed = transformed + variables['dense_bias']

        weights_skip = variables['skip']
        skip_contribution = tf.matmul(out, weights_skip[0, :, :])

        return skip_contribution, input_batch + transformed

    def _create_network(self, input_batch, global_condition_batch,
                        local_condition_batch=None):

        '''Construct the WaveNet network.'''
        outputs = []
        current_layer = input_batch

#        current_layer = self._create_causal_layer(current_layer)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    is_last_layer = layer_index == (len(self.dilations) - 1)
                    output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation,
                        global_condition_batch, local_condition_batch,
                        is_last_layer)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            variables = self.variables[TOP_NAME]['postprocessing']
            w1 = variables['postprocess1']
            w2 = variables['postprocess2']
            if self.use_biases:
                b1 = variables['postprocess1_bias']
                b2 = variables['postprocess2_bias']

            if self.histograms:
                tf.histogram_summary('postprocess1_weights', w1)
                tf.histogram_summary('postprocess2_weights', w2)
                if self.use_biases:
                    tf.histogram_summary('postprocess1_biases', b1)
                    tf.histogram_summary('postprocess2_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            if self.use_biases:
                total += b1
            transformed1 = tf.nn.elu(total) if self.elu_not_relu else \
                                tf.nn.relu(total)

            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_biases:
                conv1 += b2
            transformed2 = tf.nn.elu(conv1) if self.elu_not_relu else \
                                tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")

        return conv2

    def _create_generator(self, input_batch, global_condition_batch):
        '''Construct an efficient incremental generator.'''
        init_ops = []
        push_ops = []
        outputs = []
        current_layer = input_batch

        channels = self.softmax_channels

        q = tf.FIFOQueue(
            1,
            dtypes=tf.float32,
            shapes=(1, self.residual_channels))
        init = q.enqueue_many(
            tf.zeros((1, 1, self.residual_channels)))

        current_state = q.dequeue()
        current_layer = tf.reshape(current_layer,
                                   [-1, self.residual_channels])

        push = q.enqueue([current_layer])
        init_ops.append(init)
        push_ops.append(push)

#        current_layer = self._generator_causal_layer(
#                            current_layer, current_state)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):

                    q = tf.FIFOQueue(
                        dilation,
                        dtypes=tf.float32,
                        shapes=(1, self.residual_channels))
                    init = q.enqueue_many(
                        tf.zeros((dilation, 1, self.residual_channels)))

                    current_state = q.dequeue()
                    push = q.enqueue([current_layer])
                    init_ops.append(init)
                    push_ops.append(push)

                    output, current_layer = self._generator_dilation_layer(
                        current_layer, current_state, layer_index, dilation,
                        global_condition_batch)
                    outputs.append(output)
        self.init_ops = init_ops
        self.push_ops = push_ops

        with tf.name_scope('postprocessing'):
            variables = self.variables[TOP_NAME]['postprocessing']
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = variables['postprocess1']
            w2 = variables['postprocess2']
            if self.use_biases:
                b1 = variables['postprocess1_bias']
                b2 = variables['postprocess2_bias']

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            if self.use_biases:
                total += b1
            transformed1 = tf.nn.relu(total)
            conv1 = tf.matmul(transformed1, w1[0, :, :])
            if self.use_biases:
                conv1 += b2
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.matmul(transformed2, w2[0, :, :])

        return conv2

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.
            Args: input_batch
                  Discretized waveform as integer in range 0 to quantization
                  levels-1 (+1 if ctc_loss) as produced by mu-law encoding.
         '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=self.softmax_channels,
                dtype=tf.float32)
            encoded = tf.reshape(encoded,
                                 [1, -1, self.softmax_channels])
            return encoded

    def _embed_input(self, input_batch):
        '''Looks up the embeddings of the the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('input_embedding'):
            embedding_table = self.variables[TOP_NAME]['embeddings'][
                                             'input_embedding']
            embedding = tf.nn.embedding_lookup(embedding_table,
                                               input_batch)
            shape = [1, -1, self.residual_channels]
            embedding = tf.reshape(embedding, shape)
        return embedding

    def _embed_gc(self, global_condition):
        embedding = None
        if self.global_condition_cardinality is not None:
            # Only lookup the embedding if the global condition is presented
            # as an integer of mutually-exclusive categories ...
            embedding_table = self.variables[TOP_NAME]['embeddings'][
                                              'gc_embedding']
            embedding = tf.nn.embedding_lookup(embedding_table,
                                               global_condition)
        elif global_condition is not None:
            # ... else the global_condition (if any) is already provided
            # as an embedding.

            # In this case, the number of global_embedding channels must be
            # equal to the the last dimension of the global_condition tensor.
            gc_batch_rank = len(global_condition.get_shape)
            dims_match = (global_condition.get_shape()[gc_batch_rank - 1] ==
                          self.global_condition_channels)
            if not dims_match:
                raise ValueError('Shape of global_condition {} does not'
                                 ' match global_condition_channels {}.'.
                                 format(self.global_condition.get_shape(),
                                        self.global_condition_channels))
            embedding = global_condition

        if embedding is not None:
            embedding = tf.reshape(
                embedding, [1, 1, self.global_condition_channels])

        return embedding

    def predict_proba(self, waveform, global_condition=None,
                      local_condition=None, name='wavenet'):
        '''Computes the probability distribution of the next sample based on
        all samples in the input waveform.
        If you want to generate audio by feeding the output of the network back
        as an input, see predict_proba_incremental for a faster alternative.'''
        with tf.name_scope(name):
            encoded = self._embed_input(waveform)
            gc_embedding = self._embed_gc(global_condition)
            raw_output = self._create_network(encoded, gc_embedding,
                                              local_condition)
            out = tf.reshape(raw_output, [-1, self.softmax_channels])
            # Cast to float64 to avoid bug in TensorFlow
            if 'temperature' in self.variables[TOP_NAME]['postprocessing']:
                temp = self.variables[TOP_NAME]['postprocessing'
                            ]['temperature']
            else:
                temp = None
            proba = output_to_probs(raw_output=out, temp=temp,
                                    use_gumbel=False,
                                    gumbel_noise=False)

#            proba = tf.cast(
#                tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
            last = tf.slice(
                proba,
                [tf.shape(proba)[0] - 1, 0],
                [1, self.softmax_channels])
            return tf.reshape(last, [-1])

    def predict_proba_incremental(self, waveform, global_condition=None,
                                  name='wavenet'):
        '''Computes the probability distribution of the next sample
        incrementally, based on a single sample and all previously passed
        samples.'''
        if self.filter_width > 2:
            raise NotImplementedError("Incremental generation does not "
                                      "support filter_width > 2.")

        with tf.name_scope(name):
            encoded = self._embed_input(waveform)
            gc_embedding = self._embed_gc(global_condition)

            raw_output = self._create_generator(encoded, gc_embedding)
            out = tf.reshape(raw_output, [-1, self.softmax_channels])
            proba = tf.cast(
                tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
            last = tf.slice(
                proba,
                [tf.shape(proba)[0] - 1, 0],
                [1, self.softmax_channels])
            return tf.reshape(last, [-1])

    def _shift_one_sample(self, waveform):
        # Shift original input left by one sample, which means that
        # each output sample has to predict the next input sample.
        shifted = tf.slice(waveform, [0, 1, 0],
                           [-1, tf.shape(waveform)[1] - 1, -1])
        shifted = tf.pad(shifted, [[0, 0], [0, 1], [0, 0]])
        return shifted

    def receptive_field(self):
        max_dilation = max(self.dilations)
        num_stacks = self.dilations.count(max_dilation)
        size = max_dilation*2 + (num_stacks-1)*(max_dilation*2-1)
        return size

    def _to_sparse(self, val):
        self.labs = val
        indices = tf.where(tf.greater(val, -1))
        self.indices = indices
        sparse = tf.SparseTensor(indices=indices,
                               values=tf.gather_nd(val, indices),
                               shape=tf.cast(tf.shape(val), dtype=tf.int64))
        return sparse

    def _extend_to_match(self, target_shape, source):
        source_shape = tf.shape(source)
        source_length = source_shape[1]
        prelude_length = tf.cast((target_shape[1] - source_length) / 2, tf.int32)
        postlude_length = target_shape[1] - prelude_length - source_length
        prelude_shape = tf.cast(source_shape, tf.int32)
        prelude_shape = [source_shape[0], prelude_length, source_shape[2]]
        postlude_shape = [source_shape[0], postlude_length, source_shape[2]]
        prelude = tf.fill(postlude_shape, self.quantization_channels)
        postlude = tf.fill(postlude_shape, self.quantization_channels)
        return tf.concat(axis=1, values=[prelude, source, postlude])

    def temperature(self):
        return self.variables[TOP_NAME]['postprocessing']['temperature']

    # Returns the mean of all the orthogonal regularization losses.
    def orthog_loss(self):
        if len(self.orthogonal_reg_losses) > 0:
            orthog_loss = sum([orthog_loss for orthog_loss in
                                    self.orthogonal_reg_losses])
            orthog_loss /= tf.to_float(len(self.orthogonal_reg_losses))
        else:
            orthog_loss = None
        return orthog_loss

    def loss(self,
             input_batch,
             global_condition_batch=None,
             loss_prefix='',
             name='wavenet',
             local_condition_batch=None):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
        with tf.name_scope(name):
            gc_embedding = self._embed_gc(global_condition_batch)
            # We mu-law encode and quantize the input audioform.
            discretized_input = mu_law_encode(input_batch,
                                              self.quantization_channels)
            discretized_input = tf.reshape(discretized_input, [1, -1, 1])

            if local_condition_batch is not None:
                # Extend the discretized input to match the local conditions
                # length by "padding" with blank (per ctc loss) values.
                self.lc_shape = tf.shape(local_condition_batch)
                self.disc_input_shape = tf.shape(discretized_input)
                extended_discretized_input = self._extend_to_match(
                    target_shape=tf.shape(local_condition_batch),
                    source=discretized_input)
                self.extended_shape = tf.shape(extended_discretized_input)
            else:
                extended_discretized_input = discretized_input

            network_input = self._embed_input(extended_discretized_input)

            raw_output = self._create_network(network_input, gc_embedding,
                                              local_condition_batch)

            with tf.name_scope('loss'):
                prediction = tf.reshape(raw_output,
                                            [-1, self.softmax_channels])
                reduced_loss = 0.0
                if self.freqloss is not None:
                    # Compute loss using FrequencyDomainLoss
                    # input_batch = tf.reshape(input_batch, [1,-1,1])
                    undiscretized_input = mu_law_decode(discretized_input,
                                          self.quantization_channels)
                    undiscretized_input = tf.reshape(undiscretized_input,
                                                     [1,-1,1])
                    shifted = self._shift_one_sample(undiscretized_input)
                    temp = None
                    probs = output_to_probs(raw_output=prediction,
                                            use_gumbel = False,
                                            temp=temp)
                    waveform = self.freqloss.probs_to_waveform(probs=probs)
                    waveform = tf.reshape(waveform, [1, -1, 1])
                    reconstruction_loss = \
                        self.freqloss(target=shifted, actual=waveform)
                    # self.entropy_loss = 0.0*probs_to_entropy_bits(probs)
                    loss = reconstruction_loss
                    reduced_loss = 0.1 *tf.reduce_mean(loss)  # + self.entropy_loss

                if self.cross_entropy_loss:
                    # Use the usual cross-entropy loss.
                    one_hotted_input = self._one_hot(extended_discretized_input)
                    shifted = self._shift_one_sample(one_hotted_input)
                    loss = tf.nn.softmax_cross_entropy_with_logits(
                        logits=prediction,
                        labels=tf.reshape(shifted, [-1, self.softmax_channels]))
                    reduced_loss += tf.reduce_mean(loss)

                tf.summary.scalar(loss_prefix+'loss', reduced_loss)

                return reduced_loss
