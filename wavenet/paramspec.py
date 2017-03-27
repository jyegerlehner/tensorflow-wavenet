# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

identities = dict()

def create_variable_from_spec(param_spec):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    if param_spec.initial_value is None:
        variable = tf.Variable(initializer(shape=param_spec.shape),
                                           name=param_spec.name)
    else:
        # variable = tf.Variable(initial_value=param_spec.initial_value)
        variable = tf.constant(value=param_spec.initial_value)

    return variable


#def create_orthog_loss(param, param_name):
#    shape = tf.shape(param)
#    rank = tf.rank(param)

#    # Squash it to a rank 2 tensor (matrix).
#    reshaped = tf.reshape(param, [-1, shape[rank-1]])

#    reshaped_shape = tf.shape(reshaped)
#    prod = tf.cond(reshaped_shape[0] < reshaped_shape[1],
#                   lambda: tf.matmul(reshaped, tf.transpose(reshaped)),
#                   lambda: tf.matmul(tf.transpose(reshaped), reshaped))
#    loss = tf.nn.l2_loss(prod - tf.identity(prod),
#                         name=param_name + '_orthog_reg_loss')
#    return loss

# Create a loss for orthogonal regularization of weights.
def create_orthog_loss(param, param_name, shape):
    rank = len(shape)
    cols = shape[rank-1]
    size = 1
    for dim in shape:
        size *= dim
    rows = size // cols
    # Squash it to a rank 2 tensor (matrix).
    reshaped = tf.reshape(param, [rows, cols])
    prod_dim = rows if rows < cols else cols
    prod = tf.matmul(reshaped, tf.transpose(reshaped)) if rows < cols else \
            tf.matmul(tf.transpose(reshaped), reshaped)
    if prod_dim in identities:
        ident = identities[prod_dim]
    else:
        ident = tf.constant(value = np.identity(prod_dim), dtype=tf.float32)
        identities[prod_dim] = ident
    loss = tf.nn.l2_loss(prod - ident, name=param_name + '_orthog_reg_loss') \
            / tf.to_float(size)
    #return loss, ident, prod
    return loss


def create_embedding_table_from_spec(param_spec):
    assert len(param_spec.shape) == 2
    if param_spec.shape[0] == param_spec.shape[1]:
        # Make a one-hot encoding as the initial value.
        initial_val = np.identity(n=param_spec.shape[0], dtype=np.float32)
        return tf.Variable(initial_val, name=param_spec.name)
    else:
        initializer = tf.truncated_normal(param_spec.shape,
                                          mean=0.0,
                                          stddev=0.3,
                                          dtype=param_spec.dtype)
        variable = tf.Variable(initializer, name=param_spec.name)
        return variable

def create_bias_variable_from_spec(param_spec, value=0.0):
    name = param_spec.name
    shape = param_spec.shape
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)

def show_param_tree(tree, indent=""):
    print(indent + tree.name)
    for param in tree.params:
        print(indent + "    {}, {}, {}, {}".format(param.name,
            param.shape, param.dtype, param.kind))
    for child in tree.children:
        show_param_tree(child, indent+"  ")

class ParamTree:
    def __init__(self, name):
        # Name of this scope.
        self.name = name
        # Children ParamTree of this scope.
        self.children = []
        # List of ParamSpecs at this scope.
        self.params = []

    def add_param(self, param_spec):
        self.params.append(param_spec)

    def add_child(self, name):
        subtree = ParamTree(name=name)
        self.children.append(subtree)
        return subtree

def print_param(spec):
    print("Param:{}".format(spec.name))
    print("   Computed:{}".format(spec.computed_not_stored))
    size = 1
    for dim in spec.shape:
        size = size * dim
    print("   Size:{} floats".format(size))
    print("   Kind:{}".format(spec.kind))
    print("")


class ParamSpec:
    def __init__(self, name, shape, kind, regularization,
                  dtype=tf.float32):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.kind = kind
        self.computed_not_stored = None
        self.regularization = regularization

    def size(self):
        siz = 1
        for dim in self.shape:
            siz *= dim
        return siz

class StoredParm(ParamSpec):
    def __init__(self, name, shape, kind, regularization,
                 dtype=tf.float32,
                 initial_value=None):
        ParamSpec.__init__(self, name=name, shape=shape, kind=kind,
                           dtype=dtype,
                           regularization=regularization)
        self.computed_not_stored = False
        self.initial_value = initial_value

class ComputedParm(ParamSpec):
    def __init__(self, name, shape, kind, regularization,
                 dtype=tf.float32,
                 initial_value=None):
        ParamSpec.__init__(self, name=name, shape=shape, kind=kind,
                           dtype=dtype,
                           regularization=regularization)
        self.computed_not_stored = True
        self.initial_value=initial_value

def make_variable(param_spec):
    return create_var(name=param_spec.name,
                      shape=param_spec.shape,
                      type=param_spec.type)

variable_factory = {'filter': create_variable_from_spec,
                    'bias': create_bias_variable_from_spec,
                    'embedding': create_embedding_table_from_spec}

def create_var(param_spec):
    assert param_spec.kind in variable_factory
    return variable_factory[param_spec.kind](param_spec)


'''
Create the stored variables for a model. Stored variables are the parameters
that are stored in memory, instead of computed by another network.

Args:
  param_tree: Specifies the tree of scopes and parameters.

  parent: The parent dict to which we are adding parameters and subscopes

'''
def create_vars(spec_tree, computed_not_stored, parm_factory,
                parent=None, orthog_reg_losses=None):
    if parent is None:
        parent = dict()

    if orthog_reg_losses is None:
        orthog_reg_losses = []

    # Creation of parameters can be a two-stage process: creation of stored
    # variables might happen after creation of computed variables. In which
    # case the tree branches will already have been created.
    if spec_tree.name in parent:
        layer = spec_tree[spec_tree.name]
    else:
        layer = dict()
        parent[spec_tree.name] = layer

    with tf.variable_scope(spec_tree.name):
        # Create each parameter at this scope.
        for param_spec in spec_tree.params:
            if param_spec.computed_not_stored == computed_not_stored:
                assert param_spec.name not in layer
#                if computed_not_stored:
#                    print_param(param_spec)
                layer[param_spec.name] = parm_factory(param_spec)

                # We regularize whether-or-not the "param" is computed or
                # stored.
                if param_spec.regularization:
#                    if param_spec.kind == 'bias':
#                        reg_loss = create_bias_reg_loss(layer[param_spec.name],
#                                                        param_spec.name,
#                                                        param_spec.shape)

#                    else:
                        reg_loss = create_orthog_loss(layer[param_spec.name],
                                                         param_spec.name,
                                                         param_spec.shape)
                        orthog_reg_losses.append(reg_loss)

        for item in spec_tree.children:
            # Recursively create subtrees.
            create_vars(item,
                        computed_not_stored,
                        parm_factory,
                        layer,
                        orthog_reg_losses)

    return (parent, orthog_reg_losses)

