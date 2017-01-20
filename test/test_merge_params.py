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
                     show_params)

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
            layer_count=None,
            dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256,
                       1, 2, 4, 8, 16, 32, 64, 128, 256,
                       1, 2, 4, 8, 16, 32, 64, 128, 256],
            gated_linear=False,
            density_conditioned=False,
            compute_the_params=True,
            non_computed_params=['text_embedding'])


        input_spec = InputSpec(
            kind='quantized_scalar',
            name='sample_density',
            opts={'quant_levels':100, 'range_min':0.5, 'range_max':1.5})

        show_param_tree(self.text_encoder.param_specs)
        print("=======================================")
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

    def testParamMerge(self):
        encoder_params = self.parameter_producer.create_params(
            input_value=self.sample_density_placeholder)
        print("params before:")
        show_params(self.text_encoder.variables)
        self.text_encoder.merge_params(encoder_params)
        print("======================================")
        print("params after:")
        show_params(self.text_encoder.variables)
#        print("params after:{}".format(self.text_encoder.variables))

if __name__ == '__main__':
    tf.test.main()

