# -*- coding: utf-8 -*-

import tensorflow as tf


def create_variable_from_spec(param_spec):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=param_spec.shape),
                                       name=param_spec.name)
    return variable


def create_embedding_table_from_spec(param_spec):
    assert len(param_spec.shape) == 2
    if param_spec.shape[0] == param_spec.shape[1]:
        # Make a one-hot encoding as the initial value.
        initial_val = np.identity(n=param_spec.shape[0], dtype=param_spec.dtype)
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


class ParamSpec:
    def __init__(self, name, shape, kind, dtype=tf.float32):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.kind = kind
        self.computed_not_stored = None

    def size(self):
        siz = 1
        for dim in self.shape:
            siz *= dim
        return siz

class StoredParm(ParamSpec):
    def __init__(self, name, shape, kind, dtype=tf.float32):
        ParamSpec.__init__(self, name, shape, kind, dtype)
        self.computed_not_stored = False

class ComputedParm(ParamSpec):
    def __init__(self, name, shape, kind, dtype=tf.float32):
        ParamSpec.__init__(self, name, shape, kind, dtype)
        self.computed_not_stored = True

def make_variable(param_spec):
    return create_var(name=param_spec.name,
                      shape=param_spec.shape,
                      type=param_spec.type)

variable_factory = {'filter': create_variable_from_spec,
                    'bias':create_bias_variable_from_spec,
                    'embedding':create_embedding_table_from_spec}

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
def create_vars(spec_tree, computed_not_stored, parm_factory, parent=None):
    if parent is None:
        parent = dict()

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
                layer[param_spec.name] = parm_factory(param_spec)

        for item in spec_tree.children:
            # Recursively create subtrees.
            create_vars(item,
                        computed_not_stored,
                        parm_factory,
                        layer)

    return parent
