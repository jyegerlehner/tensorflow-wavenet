#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf


def compute_sample_density(audio, text):
    text_shape = tf.shape(text)
    text_length = tf.cast(text_shape[0], dtype=tf.float32)
    audio_length = tf.shape(audio)[0]
    # Number of samples per character.
    sample_density = tf.cast(audio_length, dtype=tf.float32) / text_length
    return (sample_density, audio_length)

