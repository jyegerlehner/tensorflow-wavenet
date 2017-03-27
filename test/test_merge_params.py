#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import random
from wavenet import (WaveNetModel, time_to_batch, batch_to_time, causal_conv,
                     optimizer_factory, mu_law_decode, ConvNetModel,
                     InputSpec, ParamProducerModel, show_param_tree,
                     show_params, quantize_value, create_embedding_table,
                     quantize_interp_embedding, create_repeated_embedding)

LAYER_COUNT = 6
TEXT_ENCODER_CHANNELS = 8
TEXT_ENCODER_OUTPUT_CHANNELS = 16 # 128 # 512
LOCAL_CONDITION_CHANNELS = 16

class TestParamMerge(tf.test.TestCase):
    def setUp(self):
        # Create text encoder network.
        self.text_encoder = ConvNetModel(
            encoder_channels=TEXT_ENCODER_CHANNELS,
            histograms=False,
            output_channels=TEXT_ENCODER_OUTPUT_CHANNELS,
            local_condition_channels=LOCAL_CONDITION_CHANNELS,
            elu_not_relu=True,
            layer_count=None,
            dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256,
                       1, 2, 4, 8, 16, 32, 64, 128, 256,
                       1, 2, 4, 8, 16, 32, 64, 128, 256],
            gated_linear=False,
            density_options=None,
            compute_the_params=True,
            non_computed_params=['text_embedding'])

        input_spec = InputSpec(
            kind='quantized_scalar',
            name='sample_density',
            opts={'quant_levels':100, 'range_min':0.5, 'range_max':1.5})

        # show_param_tree(self.text_encoder.param_specs)
        # print("=======================================")
        self.parameter_producer = ParamProducerModel(
            input_spec=input_spec,
            output_specs=self.text_encoder.param_specs,
            residual_channels=128)


        self.audio_placeholder = tf.placeholder(dtype=tf.float32)
        self.gc_placeholder = tf.placeholder(dtype=tf.int32)
        self.ascii_placeholder = tf.placeholder(dtype=tf.int32)
        self.lc_placeholder = tf.placeholder(dtype=tf.float32)
        self.samples_placeholder = tf.placeholder(dtype=tf.int32)
        self.sample_density_placeholder = tf.placeholder(dtype=tf.float32,
                                                         shape=[1,1])

#    def testParamMerge(self):
#        encoder_params = self.parameter_producer.create_params(
#            input_value=self.sample_density_placeholder)
#        print("============================================")
#        print("params before:")
#        show_params(self.text_encoder.variables)

#        self.text_encoder.merge_params(encoder_params)

#        print("======================================")
#        print("params after:")
#        show_params(self.text_encoder.variables)


    def testInterpolatedEmbedding(self):

        QUANT_LEVELS = 3
        channels = 4
        np_table = np.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        table_shape = [QUANT_LEVELS+1, channels]

        table = tf.constant(value=np_table,
                            shape=table_shape,
                            verify_shape=True)

        # np_vals = np.array([0.1*i for i in range(60)], dtype=np.float32)
        np_vals = np.array([0.0, 15.0, 20.0, 50.0, 60.0, 200.0], dtype=np.float32)

        interpolated_embedding = quantize_interp_embedding(
                                    value=tf.constant(np_vals),
                                    quant_levels=QUANT_LEVELS,
                                    min=0.0,
                                    max=60.0,
                                    embedding_table=table)

        with self.test_session() as sess:
            embedding = sess.run(interpolated_embedding)

        expected_embedding = np.array([[1.00, 0.00, 0.0, 0.0],
                                       [0.25, 0.75, 0.0, 0.0],
                                       [0.00, 1.00, 0.0, 0.0],
                                       [0.00, 0.00, 0.5, 0.5],
                                       [0.00, 0.00, 0.0, 1.0],
                                       [0.00, 0.00, 0.0, 1.0]])
#        print("embedding:{}".format(embedding))
#        print("expected embedding:{}".format(expected_embedding))
#        print("lbv:{} ubv:{} ratio:{} lower ratio:{}".format(lbv, ubv, rv, lrv))

        np.testing.assert_allclose(embedding, expected_embedding, rtol=0.001)


    def testUniformInit(self):
        embedding_size = 5
        table_entries = 3
        initializer_shape = [1,embedding_size]
        table_shape = [table_entries, embedding_size]
        table = create_repeated_embedding(name="atable",
                                       shape=table_shape)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            table_val = sess.run(table)

        first_embedding_val = table_val[0, :]
        ctr = 0
        for entry in range(table_val.shape[0]):
            # Each embedding should merely repeat the first embedding
            np.testing.assert_allclose(table_val[entry,:],
                                       first_embedding_val, rtol=0.001)
            ctr = ctr + 1

        assert ctr > 1

#    def testOrthogonalLoss(self):
#        param_shape = [4,8]
#        initializer=tf.contrib.layers.xavier_initializer_conv2d()
#        param = tf.Variable(initializer(param_shape))
#        init = tf.global_variables_initializer()
#        prod = tf.matmul(param, tf.transpose(param))

#        (loss_a, ident_a, prod_a) = create_orthog_loss(param, "dummy",
#                                                       param_shape)

#        with self.test_session() as sess:
#            sess.run(init)
#            [param_val, prod_val, loss_val, ident_val, prod_a_val] = sess.run([param, prod, loss_a, ident_a, prod_a])
#            print("param_val:{}".format(param_val))
#            print("=============================================")
#            print("prod:{}".format(prod_val))
#            print("=============================================")
#            print("loss_val:{}".format(loss_val))
#            print("=============================================")
#            print("ident_val:{}".format(ident_val))
#            print("=============================================")
#            print("prod_a_val:{}".format(prod_a_val))










if __name__ == '__main__':
    tf.test.main()

