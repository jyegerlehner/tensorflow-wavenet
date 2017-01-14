# -*- coding: utf-8 -*-
import tensorflow as tf

class InputSpec:
    '''
    Defines an input to a ParamCreatorMLP.

    Args:
        kind: Either 'vector' for numerical valued vector,
              'quantized_scalar' for a scalar that is to be quantized into a
              finite number of ranges, or 'category', for an integer
              specifying which of a number of categories.

        name: The name of the input.

        size: This is the size of the embedding vector if the input kind is
              either 'quantized_scalar' or 'category', or number of channels

        cardinality: The number of quanitization levels, if the input is
                    'quantized_scalar', or number of categories, if the
                    input kind is 'category'. Otherwise must be none.

    '''
    def __init__(self, kind, name, opts):
        assert kind in {'vector', 'quantized_scalar', 'category'}
        if kind == 'quantized_scalar':
            assert 'quant_levels' in opts
            assert 'range_min' in opts
            assert 'range_max' in opts
        elif kind == 'category':
            assert 'cardinality' in opts
        elif kind == 'vector':
            assert 'size' in opts
        else:
            assert False

        assert name is not None
        assert len(name) > 0

        self.kind = kind
        self.name = name
        self.opts = opts



