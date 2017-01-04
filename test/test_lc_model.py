""" Unit tests for a locally-conditioned WaveNet model that check that it can
    train on audio data."""
import sys
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import random
from wavenet import (WaveNetModel, time_to_batch, batch_to_time, causal_conv,
                     optimizer_factory, mu_law_decode, ConvNetModel)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

SAMPLE_RATE_HZ = 2000.0  # Hz
SAMPLE_DURATION = 0.5  # Seconds
SAMPLE_PERIOD_SECS = 1.0 / SAMPLE_RATE_HZ
MOMENTUM = 0.95
QUANTIZATION_CHANNELS = 256
NUM_SPEAKERS = 3
RECEPTIVE_FIELD = 256
F1 = 155.56  # E-flat frequency in hz
F2 = 196.00  # G frequency in hz
F3 = 233.08  # B-flat frequency in hz
# This is set to the upsample rate of the conv net, and thus should be
# the upper bound on how many audio samples correspond to a single
# character. The CTC loss can create blanks in order allow the text
# to catch up to the audio.
UPSAMPLE_RATE = 200
MEDIAN_SAMPLES_PER_CHAR = 200
LAYER_COUNT = 6
TEXT_ENCODER_CHANNELS = 8
TEXT_ENCODER_OUTPUT_CHANNELS = 16 # 128 # 512
LOCAL_CONDITION_CHANNELS = 16

def ascii_to_text(ascii):
    return [chr(code) for code in ascii]

def find_nearest(freqs, power_spectrum, frequency):
    # Return the power of the bin nearest to the target frequency.
    index = (np.abs(freqs - frequency)).argmin()
    return power_spectrum[index]


def check_waveform(assertion, generated_waveform, gc_category):
    librosa.output.write_wav('/tmp/sine_test{}.wav'.format(gc_category),
                             generated_waveform,
                             SAMPLE_RATE_HZ)
    power_spectrum = np.abs(np.fft.fft(generated_waveform))**2
    freqs = np.fft.fftfreq(generated_waveform.size, SAMPLE_PERIOD_SECS)
    indices = np.argsort(freqs)
    indices = [index for index in indices if freqs[index] >= 0 and
               freqs[index] <= 500.0]
    power_spectrum = power_spectrum[indices]
    freqs = freqs[indices]
#    plt.plot(freqs[indices], power_spectrum[indices])
#    plt.show()
    power_sum = np.sum(power_spectrum)
    f1_power = find_nearest(freqs, power_spectrum, F1)
    f2_power = find_nearest(freqs, power_spectrum, F2)
    f3_power = find_nearest(freqs, power_spectrum, F3)
    if gc_category is None:
        # We are not globally conditioning to select one of the three sine
        # waves, so expect it across all three.
        expected_power = f1_power + f2_power + f3_power
        assertion(expected_power, 0.7 * power_sum)
    else:
        # We expect spectral power at the selected frequency
        # corresponding to the gc_category to be much higher than at the other
        # two frequencies.
        frequency_lut = {0: f1_power, 1: f2_power, 2: f3_power}
        other_freqs_lut = {0: f2_power + f3_power,
                           1: f1_power + f3_power,
                           2: f1_power + f2_power}
        expected_power = frequency_lut[gc_category]
        # Power at the selected frequency should be at least 10 times greater
        # than at other frequences.
        # This is a weak criterion, but still detects implementation errors
        # in the code.
        assertion(expected_power, 10.0*other_freqs_lut[gc_category])

    # print("gc category:", gc_category)
    # print("Power sum {}, F1 power:{}, F2 power:{}, F3 power:{}".
    #        format(power_sum, f1_power, f2_power, f3_power))
def char_to_index(a_char):
    return {
        'a': 0,
        'b': 1,
        'c': 2,
        ' ': 3
        }[a_char]

def char_to_asc(some_text):
    ascii_seq = [ord(a_char) for a_char in some_text]
    return ascii_seq

class TestLCNet(tf.test.TestCase):
    def setUp(self):
        print('TestNetWithLocalConditioning setup.')
        sys.stdout.flush()

        self.optimizer_type = 'adam'
        self.learning_rate = 0.0004
        self.generate = True
        self.momentum = 0.9
        self.global_conditioning = False
        self.train_iters = 100
        self.net = WaveNetModel(
            dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256,
                       1, 2, 4, 8, 16, 32, 64, 128, 256,
                       1, 2, 4, 8, 16, 32, 64, 128, 256],
            filter_width=2,
            residual_channels=16,
            dilation_channels=16,
            quantization_channels=QUANTIZATION_CHANNELS,
            use_biases=True,
            skip_channels=256,
            local_condition_channels=LOCAL_CONDITION_CHANNELS,
            ctc_loss=False,
            gated_linear=False)

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
            density_conditioned=True)

        self.audio_placeholder = tf.placeholder(dtype=tf.float32)
        self.gc_placeholder = tf.placeholder(dtype=tf.int32)  \
            if self.global_conditioning else None
        self.ascii_placeholder = tf.placeholder(dtype=tf.int32)
        self.lc_placeholder = tf.placeholder(dtype=tf.float32)
        self.samples_placeholder = tf.placeholder(dtype=tf.int32)
        self._initialize_source_waveforms()
        self.last_start_times = {0:0, 1:0, 2:0, 3:0}



    def _save_net(self, sess):
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.save(sess, '/tmp/test_lc.ckpt')

    def _load_net(self, sess):
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.restore(sess, '/tmp/test_lc.ckpt')

    def generate_waveform(self, sess, fast_generation, gc, next_sample_probs,
                          local_conditions, ascii_sequence, index):
        receptive_field = self.net.receptive_field()
        results = []
        samples_to_generate = (len(ascii_sequence) -
                               self.net.receptive_field() + 1) * UPSAMPLE_RATE


        # Generate the local conditions from this text.
        lc = sess.run(local_conditions,
                      feed_dict={self.ascii_placeholder: ascii_sequence})
        samples_to_generate = lc.shape[1]
        waveform = [128]*samples_to_generate
        waveform = np.array(waveform)
        print("Samples to generate:{}, lc shape:{} for text:{}".format(
            samples_to_generate,lc.shape, ascii_to_text(ascii_sequence)))

        for i in range(1,samples_to_generate):
            if i % 100 == 0:
                print("Generating {} of {} for {}.".format(i,
                    samples_to_generate, ascii_to_text(ascii_sequence)))
                sys.stdout.flush()
#            if fast_generation:
#                window = waveform[-1]
#            else:
#                if len(waveform) > receptive_field:
#                    window = waveform[-RECEPTIVE_FIELD:]
#                else:
#                    window = waveform
            if i >= receptive_field:
                window = waveform[i-receptive_field:i]
                lc_window = lc[:,i-receptive_field:i,:]
            else:
                window = waveform[:i]
                lc_window = lc[:,:i,:]

            # Run the WaveNet to predict the next sample.
            feed_dict = {self.samples_placeholder: window,
                         self.lc_placeholder: lc_window}
            if gc is not None:
                feed_dict[gc_placeholder] = gc
            results = sess.run([next_sample_probs], feed_dict=feed_dict)

            sample = np.random.choice(
               np.arange(results[0].shape[0]), p=results[0])
            waveform[i] = sample

        # Skip the first number of samples equal to the size of the receptive
        # field.
        #waveform = waveform[RECEPTIVE_FIELD:]
        decode = mu_law_decode(self.samples_placeholder, QUANTIZATION_CHANNELS)
        decoded_waveform = sess.run(decode,
             feed_dict={self.samples_placeholder: waveform})
        # Strip the blank entries.
        print("Decoded waveform shape before stripping blank:{}".format(
              decoded_waveform.shape))
        non_blank = decoded_waveform <= 1.0
        decoded_waveform = decoded_waveform[non_blank]
        print("Decoded waveform shape after stripping blank:{}".format(
              decoded_waveform.shape))
        librosa.output.write_wav('/tmp/lc_test{}.wav'.format(index),
                                 decoded_waveform, SAMPLE_RATE_HZ)

        return decoded_waveform


    def generate_waveforms(self, sess, net, fast_generation, global_condition, ascii):
        next_sample_probs = net.predict_proba(self.samples_placeholder,
            self.gc_placeholder, local_condition=self.lc_placeholder)
        local_conditions = self.text_encoder.upsample(self.ascii_placeholder,
                             tf.shape(self.ascii_placeholder)[0]*UPSAMPLE_RATE)

        num_waveforms = len(ascii)
        gc = None
        waveforms = [None] * num_waveforms
        for waveform_index in range(num_waveforms):
            if global_condition is not None:
                gc = global_condition[waveform_index, :]
            # Generate a waveform for each speaker id.
            waveforms[waveform_index] = self.generate_waveform(
                sess, fast_generation, gc, next_sample_probs, local_conditions,
                ascii[waveform_index], waveform_index)

        return waveforms, global_condition


    def interpolate(self, waveform, sample_start, ramp_in_end, sample_end,
                    prev_char, curr_char, source_amplitudes):

        prev_inx = char_to_index(prev_char)
        curr_inx = char_to_index(curr_char)

        for index in range(sample_start, ramp_in_end):
            curr_weight = (index - sample_start) / float(ramp_in_end -
                                                    sample_start)

            prev_weight = 1.0 - curr_weight
            assert (curr_weight >= 0.0) and (curr_weight <= 1.0)
            assert (prev_weight >= 0.0) and (prev_weight <= 1.0)
            assert (curr_weight + prev_weight == 1.0)
            curr_char_start = self.last_start_times[curr_inx]
            prev_char_start = self.last_start_times[prev_inx]
            weighted_val = curr_weight *  \
                  source_amplitudes[curr_inx, index - curr_char_start] \
                 + prev_weight * source_amplitudes[prev_inx,  \
                                        index - prev_char_start]
            waveform.append(weighted_val)
        return waveform

    prev_char = None
    start_sample = 0
    def make_training_waveform(self, text_sequence, duration_ratio=1.0):

        # duration of each char in samples.
        char_duration = MEDIAN_SAMPLES_PER_CHAR
        char_duration = int(char_duration * duration_ratio)

        # initialize all weights to zero.
        weights = np.zeros(shape=(4, len(text_sequence) * char_duration))

        char_pairs = []
        for index in range(len(text_sequence)):
            curr_char = text_sequence[index]
            if index == 0:
                prev_char = ' '
            else:
                prev_char = text_sequence[index-1]
            char_pairs.append((prev_char, curr_char))

        waveform = []
        sample_start = 0

        # Spend 20% of the character's time ramping from the previous
        # character's waveform to the current character's.
        ramp_duration = int(0.2 * char_duration)
        while char_duration % 2 != 0:
            char_duration -= 1

        for (prev_char, curr_char) in char_pairs:
            sample_end = sample_start + char_duration
            ramp_in_end = sample_start + ramp_duration

            char_inx = char_to_index(curr_char)
            # curr char just started at sample_start
            if (prev_char != curr_char):
                self.last_start_times[char_inx] = sample_start

            # add curr_char's contribution to the waveform.

            waveform = self.interpolate(waveform, sample_start, ramp_in_end,
                                   sample_end, prev_char, curr_char,
                                   self._source_amps)

            for index in range(ramp_in_end, sample_end):
                if index < self._source_amps.shape[1]:
                    char_start_time = self.last_start_times[char_inx]
                    val = self._source_amps[char_inx, index - char_start_time]
                else:
                    val = 0.0
                waveform.append(val)
            # Start of next char picks up at end of its previous char.
            sample_start = sample_end

        return waveform

    def _make_text_sequence(self):
        num_words = np.random.randint(low=2, high=6)
        chars = [' ']
        for wordindex in range(num_words):
            char_inx = np.random.randint(low=0, high=3)
            char = { 0:'a', 1:'b', 2:'c' }[char_inx]
            word_len = np.random.randint(low=1, high=3)
            for i in range(word_len):
                chars.append(char)

            num_spaces = np.random.randint(low=1, high=3)
            for i in range(num_spaces):
                chars.append(' ')
        return ''.join(chars)

    def _initialize_source_waveforms(self):
        sample_period = 1.0/SAMPLE_RATE_HZ
        longest_text_length = 30
        generation_seconds = longest_text_length * UPSAMPLE_RATE * \
                             sample_period
        largest_duration_ratio = 1.4
        max_generation_samples = generation_seconds * SAMPLE_RATE_HZ

        # Create the time sequence that corresponds to the longest sample
        # we will generate.
        times = np.arange(0.0, generation_seconds, sample_period)
        self._times = times

        if self.global_conditioning:
            raise ValueError("Global conditioning not supported.")

        self._source_amps = np.zeros(shape=(4, len(times)))

        self._source_amps[0,:] = np.sin(times * 2.0 * np.pi * F1) * 0.9
        self._source_amps[1,:] = np.sin(times * 2.0 * np.pi * F2) * 0.8
        self._source_amps[2,:] = np.sin(times * 2.0 * np.pi * F3) * 0.7
        # silence is zero everywhere.
        self._source_amps[3,:] = times * 0.0

#    def _pad_text(self, text):
#        # four spaces before and after to correspond to the width of the
#        # of the receptive field (8).
#        return ascii_to_text([0]*3 + char_to_asc(text) + [0]*4)


    def _make_training_pair(self):
         text = self._make_text_sequence()
         duration_ratio = 0.8*np.random.random_sample() + 0.6
         waveform = self.make_training_waveform(text,
                                                duration_ratio=duration_ratio)
#         padded_text = self._pad_text(text)
#         padded_ascii = char_to_asc(padded_text)
         ascii = char_to_asc(text)
         return (waveform, None, ascii, duration_ratio)


    def _make_training_data(self, duration_ratios=None):
        """Creates a time-series of sinusoidal audio amplitudes."""

        text_sequences = [' a b c  ',
                          ' cc b c a  c ',
                          ' a  a ',
                          ' b  a b c ',
                          ' a bb a b b c ',
                          ' c bb c a a a',
                          ' c  bb c aa a ',
                          ' c bb c aaa a',
                          ' ccc b a c a ']

        if duration_ratios is None:
            duration_ratios = []
            for i in range(len(text_sequences)):
                ratio = 0.8*np.random.random_sample() + 0.6
                duration_ratios.append(ratio)


        #duration_ratios = [0.9, 1.0, 0.8, 1.4, 0.6, 1.3, 1.1, 0.75, 1.2]

        target_amplitudes = []
        for (text_sequence, duration_ratio) in zip(text_sequences,
                                                   duration_ratios):
            waveform = self.make_training_waveform(text_sequence,
                                                   duration_ratio)
            target_amplitudes.append(waveform)

        speaker_ids = None
        ascii_sequences = [char_to_asc(text_seq) for text_seq in
                           text_sequences]

        return (target_amplitudes, speaker_ids, ascii_sequences)


    # Train a net on a short clip of 3 sine waves superimposed
    # (an e-flat chord).
    #
    # Presumably it can overfit to such a simple signal. This test serves
    # as a smoke test where we just check that it runs end-to-end during
    # training, and learns this waveform.

    def testEndToEndTraining(self):
        np.random.seed(42)
        (audio, speaker_ids, ascii) = self._make_training_data()


        i = 0
        for waveform in audio:
#            plt.plot(np.array(waveform))
#            plt.show()
            print("waveform length:{}".format(len(waveform)))
            librosa.output.write_wav('/tmp/lc_train{}.wav'.format(i),
                                     np.array(waveform, dtype=np.float32),
                                     SAMPLE_RATE_HZ)
            i += 1


#        if self.generate:
#            if len(audio.shape) == 2:
#                for i in range(audio.shape[0]):
#                    librosa.output.write_wav(
#                          '/tmp/sine_train{}.wav'.format(i), audio[i,:],
#                          SAMPLE_RATE_HZ)
#                    power_spectrum = np.abs(np.fft.fft(audio[i,:]))**2
#                    freqs = np.fft.fftfreq(audio[i,:].size,
#                                           SAMPLE_PERIOD_SECS)
#                    indices = np.argsort(freqs)
#                    indices = [index for index in indices if
#                                 freqs[index] >= 0 and
#                                 freqs[index] <= 500.0]
#                    plt.plot(freqs[indices], power_spectrum[indices])
#                    plt.show()

        local_conditions = self.text_encoder.upsample(self.ascii_placeholder,
                               tf.shape(self.audio_placeholder)[0])
        loss = self.net.loss(input_batch=self.audio_placeholder,
                             global_condition_batch=self.gc_placeholder,
                             local_condition_batch=local_conditions)
        optimizer = optimizer_factory[self.optimizer_type](
                      learning_rate=self.learning_rate, momentum=self.momentum)
        trainable = tf.trainable_variables()
        #optim = optimizer.minimize(loss, var_list=trainable)
        # clipped gradients:
        grads_and_vars = optimizer.compute_gradients(loss, var_list=trainable)
        def ClipIfNotNone(grad):
                 if grad is None:
                     return grad
                 return tf.clip_by_value(grad, -0.3, 0.3)

        capped_grads_and_vars = [(ClipIfNotNone(gv[0]), gv[1]) for gv in grads_and_vars]
        def Norm(var):
            return tf.sqrt(tf.reduce_sum(tf.square(var)))
        var_norms = [Norm(gv[1]) for gv in grads_and_vars]
        optim = optimizer.apply_gradients(capped_grads_and_vars)

        init = tf.initialize_all_variables()

        generated_waveform = None
        max_allowed_loss = 0.1
        loss_val = max_allowed_loss
        initial_loss = None
        operations = [loss, optim]
        # operations.extend(var_norms)

        if speaker_ids is not None:
            feed_dict[self.gc_placeholder] = speaker_ids

        with self.test_session() as sess:
            sess.run(init)

#            ops = []
#            ops.extend(self.text_encoder.skip_cuts)
#            (audio, speaker_ids, ascii) = self._make_training_data()
#            feed_dict = {self.audio_placeholder: audio[2],
#                         self.ascii_placeholder: ascii[2]}
#            print("text:{}".format(ascii[2]))
#            results = sess.run(self.text_encoder.output_shapes, feed_dict)
#            for result in results:
#                print("shape: {}".format(result))
#            results = sess.run(self.text_encoder.skip_cuts, feed_dict)
#            for result in results:
#                print("shape: {}".format(result))
#            result = sess.run(self.text_encoder.output_width, feed_dict)
#            print("output_width:{}".format(result))

#            result = sess.run(self.text_encoder.text_shape, feed_dict)
#            print("text_shape:{}".format(result))
#            result = sess.run(self.text_encoder.embedding_shape, feed_dict)
#            print("embedding_shape:{}".format(result))


            (audio, speaker_ids, ascii, duration_ratio) = \
                                            self._make_training_pair()
            feed_dict = {self.audio_placeholder: audio,
                         self.ascii_placeholder: ascii}

            initial_loss = sess.run(loss, feed_dict=feed_dict)

            for i in range(self.train_iters):
#                if i % len(audio) == 0:
#                    (audio, speaker_ids, ascii) = self._make_training_data()

                (audio, speaker_ids, ascii, duration_ratio) =  \
                    self._make_training_pair()
                # Rotate through each input/target-output-pair.
                feed_dict = {self.audio_placeholder: audio,
                             self.ascii_placeholder: ascii}

                results = sess.run(operations, feed_dict=feed_dict)

                if i % 10 == 0:
                    print("i: %d loss: %f, text: %s, duration_ratio: %s" % \
                        (i, results[0], ascii, duration_ratio))
#                    for result in results:
#                        print("Result:{}".format(result))

            self._save_net(sess)
#            loss_val = results[0]

#            # Sanity check the initial loss was larger.
#            self.assertGreater(initial_loss, max_allowed_loss)

#            # Loss after training should be small.
#            self.assertLess(loss_val, max_allowed_loss)

#            # Loss should be at least two orders of magnitude better
#            # than before training.
#            self.assertLess(loss_val / initial_loss, 0.02)

            if self.generate:
                if self.global_conditioning:
                    # Check non-fast-generated waveform.
                    generated_waveforms, ids = self.generate_waveforms(
                        sess, self.net, False, speaker_ids)
                    for (waveform, id) in zip(generated_waveforms, ids):
                        check_waveform(self.assertGreater, waveform, id[0])

                    # Check fast-generated wveform.
                    # generated_waveforms, ids = generate_waveforms(sess,
                    #     self.net, True, speaker_ids)
                    # for (waveform, id) in zip(generated_waveforms, ids):
                    #     print("Checking fast wf for id{}".format(id[0]))
                    #     check_waveform( self.assertGreater, waveform, id[0])

                else:
                    duration_ratios = [0.9, 1.0, 0.8, 1.4, 0.6, 1.3,
                                       1.1, 0.75, 1.2]
                    (audio, speaker_ids, ascii) = self._make_training_data()
                    #    duration_ratios=duration_ratios)
                    # Check non-incremental generation
                    # self._load_net(sess)
                    generated_waveforms, _ = self.generate_waveforms(
                        sess, self.net, False, None, ascii)
                    check_waveform(
                        self.assertGreater, generated_waveforms[0], None)


#class TestDebug(tf.test.TestCase):
#    def setUp(self):
#        # Create text encoder network.
#        self.text_encoder = ConvNetModel(
#            batch_size=BATCH_SIZE,
#            encoder_channels=TEXT_ENCODER_CHANNELS,
#            histograms=False,
#            output_channels=TEXT_ENCODER_OUTPUT_CHANNELS,
#            local_condition_channels=LOCAL_CONDITION_CHANNELS,
#            upsample_rate=UPSAMPLE_RATE,
#            layer_count=LAYER_COUNT)

#    def testDebug(self):
#        with self.test_session() as sess:
#            sess.run(tf.initialize_all_variables())
##            output_width:9
##            skip_cuts:[0, -2, -3]
##            output_shapes:[array([ 1,  8, 48], dtype=int32), array([ 1,  4, 48], dtype=int32), array([ 1,  2, 48], dtype=int32)]
#            .
#            print("=========================================================")
#            print("=========================================================")
#            print("=========================================================")
#            print("=========================================================")
#            ascii = char_to_asc('     a b c      ')
#            ascii = np.array(ascii)
#            ascii = np.reshape(ascii,(1,ascii.shape[0]))
#            upsample = self.text_encoder.upsample(ascii)

#            output_width = sess.run(self.text_encoder.output_width)
#            print('output_width:{}'.format(output_width))
#            skip_cuts = sess.run(self.text_encoder.skip_cuts)
#            print('skip_cuts:{}'.format(skip_cuts))
#            output_shapes = sess.run(self.text_encoder.output_shapes)
#            print('output_shapes:{}'.format(output_shapes))


#class TestDebugIndices(tf.test.TestCase):
#    def setUp(self):
#        print('TestDebug setup.')
#        sys.stdout.flush()

#        self.net = WaveNetModel(
#            batch_size=BATCH_SIZE,
#            dilations=[1, 2, 4, 8, 16, 32, 64,
#                       1, 2, 4, 8, 16, 32, 64],
#            filter_width=2,
#            residual_channels=32,
#            dilation_channels=32,
#            quantization_channels=QUANTIZATION_CHANNELS,
#            use_biases=True,
#            skip_channels=256,
#            local_condition_channels=LOCAL_CONDITION_CHANNELS,
#            ctc_loss=True)

#    def testDebug(self):
#        a = np.zeros([1,5])
#        ap = tf.placeholder(dtype=tf.float32)
#        b = tf.greater(ap, -1)
#        indices = tf.where(b)
#        sparse = self.net._to_sparse(a)

#        with self.test_session() as sess:
#            sess.run(tf.initialize_all_variables())
#            ops = [b, indices, sparse, ap]
#            results = sess.run(ops, {ap:a})
#            print("a:{}".format(results[3]))
#            print("b:{}".format(results[0]))
#            print("indices:{}".format(results[1]))
#            print("sparse:{}".format(results[2]))


#class TestBlankStripping(tf.test.TestCase):
#    def setUp(self):
#        self.samples_placeholder = tf.placeholder(tf.int32)

#    def testBlankStrip(self):
#        waveform = np.array([244, 254, 255, 256, 256, 128])
#        decode = mu_law_decode(self.samples_placeholder, QUANTIZATION_CHANNELS)
#        with self.test_session() as sess:
#            decoded_waveform = sess.run(decode,
#                 feed_dict={self.samples_placeholder: waveform})
#        # Strip the blank entries.
#        print("Decoded waveform shape before stripping blank:{}".format(
#              decoded_waveform.shape))

#        print("decoded waveform with blanks:{}".format(decoded_waveform))

#        non_blank = decoded_waveform <= 1.0
#        decoded_waveform = decoded_waveform[non_blank]
#        print("Decoded waveform shape after stripping blank:{}".format(
#              decoded_waveform.shape))
#        print("decoded waveform:{}".format(decoded_waveform))


if __name__ == '__main__':
    tf.test.main()
