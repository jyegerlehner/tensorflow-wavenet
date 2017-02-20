from __future__ import print_function
import tensorflow as tf

from .paramspec import (create_vars, StoredParm, ComputedParm,
                        ParamTree, create_var)

from .ops import (quantize_value, create_variable, create_embedding_table,
                  create_bias_variable, gated_residual_layer, shape_size,
                  quantize_interp_embedding, create_repeated_embedding)

# This is used for the size of the representation vector in the net,
# including the input's embedding size all the way up until the
# output-specific layers.
OUT_SPECIFIC_LAYER_COUNT = 4
COMMON_LAYER_COUNT = 0


'''
Functor to produce the output that servces as a parameter tensor in
another network.
'''
class ParamFactory:
    def __init__(self, input):
        self.input = input

    '''
    Given a param spec, produce an output matching the spec.
    '''
    def  __call__(self, param_spec):

        current = self.input
        for i in range(OUT_SPECIFIC_LAYER_COUNT):
            layer_name = param_spec.name + "_layer{}".format(i)
            current = gated_residual_layer(current,
                                           layer_name)

        # Project from channels of the input to the required channels
        # for the parameter.
        proj_weights = create_variable(name="project",
            shape=[shape_size(self.input.get_shape()), param_spec.size()])
        current = tf.matmul(current, proj_weights)

        # We created it as a flat vector; reshape to desired shape.
        current = tf.reshape(current, param_spec.shape)
        return current


class ParamProducerModel:
    '''
    An MLP that produces as outputs the parameters of another network.

    Args:
        input_spec: InputSpec describing what this takes as input.

        output_specs: A ParamSpec tree that comprise the set of parameters
                      to be created as outputs of this net.

    '''
    def __init__(self, input_spec, output_specs, residual_channels,
                 name='param_producer'):
        self.output_specs = output_specs
        self.input_spec = input_spec
        self.name = name
        self.residual_channels = residual_channels
        self.encoded_input = None
        self.lb = None
        self.ub = None
        self.interp_ratio = None

#    def _embed_density(self, sample_density):
#        quantized_density = quantize_sample_density(
#            sample_density, DENSITY_QUANT_LEVELS)

##        initializer = tf.truncated_normal(shape, mean=0.0, stddev=0.3,
##            dtype=tf.float32)
#        shape = [DENSITY_QUANT_LEVELS, self.residual_channels]
#        table = create_embedding_table('embedding', shape)

#        density_embedding = tf.nn.embedding_lookup(table,
#                                                   quantized_density)
#        shape = [1, 1, self.encoder_channels]
#        density_embedding = tf.reshape(density_embedding, shape)

#        return density_embedding

    def _common_layers(self, middle_rep):
        current=middle_rep
        for i in range(COMMON_LAYER_COUNT):
            layer_name = "common_layer{}".format(i)
            current = gated_residual_layer(current, layer_name)
        return current


    def _encode_scalar(self, input_value):
        with tf.variable_scope(self.input_spec.name):
            weights_shape = [self.residual_channels, self.input_spec.size]
            weights = create_variable('weights', weights_shape)
            # Project the input to the number of middle encoding (residual)
            # channels.
            return tf.matmul(input_value, weights)


    def _encode_quantized_scalar(self, input_value):
        with tf.variable_scope(self.input_spec.name):
            if len(input_value.get_shape()) == 0:
                input_value = tf.reshape(input_value, [1])
            quant_levels = self.input_spec.opts['quant_levels']
            table_shape = [quant_levels+1, self.residual_channels]
            table = create_repeated_embedding(
                            name=self.input_spec.name+"_embedding_table",
                            shape=table_shape)
            vect = quantize_interp_embedding(
                                    value=input_value,
                                    quant_levels=quant_levels,
                                    min=self.input_spec.opts['range_min'],
                                    max=self.input_spec.opts['range_max'],
                                    embedding_table=table)
            vect = tf.reshape(vect, [1, -1])
            return vect


    def _encode_category(self, input_value):
        with tf.variable_scope(self.input_spec.name):
            table_shape = [self.input_spec.opts['cardinality'],
                           self.residual_channels]
            table = create_embedding_table(
                            name=self.input_spec.name+"_embedding_table",
                            shape = table_shape)
            #return tf.gather_nd(table, input_value)
            vect = tf.nn.embedding_lookup(table, input_value)
            vect = tf.reshape(vect, [1, -1])
            return vect


    def _encode_input(self, input_value):
        with tf.variable_scope('inputs'):
            input_encoders = {'scalar': self._encode_scalar,
                              'quantized_scalar': self._encode_quantized_scalar,
                              'category': self._encode_category}
            return input_encoders[self.input_spec.kind](input_value)


    '''
    Create the recursive dicts containing the computed parameters
    that will be passed to receiving network.

    '''
    def create_params(self, input_value):
        with tf.variable_scope('param_producer'):
            middle_representation = self._encode_input(input_value)
            self.encoded_input = middle_representation
            middle_representation = self._common_layers(middle_representation)

            param_factory = ParamFactory(middle_representation)

            # Create the recursive dict-of-dicts of tensors that the
            # receiving network (the net whose parameters this net is
            # computing.
            outputs = create_vars(spec_tree=self.output_specs,
                                  computed_not_stored=True,
                                  parm_factory=param_factory)


            return outputs


