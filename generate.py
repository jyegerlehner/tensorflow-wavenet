from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os

import librosa
import numpy as np
import tensorflow as tf

from wavenet import (WaveNetModel, mu_law_decode, mu_law_encode, audio_reader,
                     ConvNetModel)

SAMPLES = 16000
TEMPERATURE = 1.0
LOGDIR = './logdir'
WINDOW = 8000
WAVENET_PARAMS = './wavenet_params.json'
ENCODER_PARAMS = './encoder_params.json'
SAVE_EVERY = None
SILENCE_THRESHOLD = 0.1
#ENCODER_CHANNELS = 48
#ENCODER_OUTPUT_CHANNELS = 512
#LOCAL_CONDITION_CHANNELS = 32
#UPSAMPLE_RATE = 1000  # Typical number of audio samples per
DURATION_RATIO = 1.0


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError('Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--duration_ratio',
        type=_ensure_positive_float,
        default=DURATION_RATIO,
        help='The ratio of the duration of the audio to the duration it would '
        'have if it were the median ratio of duration to number of text '
        ' characters.')
    parser.add_argument(
        '--temperature',
        type=_ensure_positive_float,
        default=TEMPERATURE,
        help='Sampling temperature')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--window',
        type=int,
        default=WINDOW,
        help='The number of past samples to take into '
        'account at each step')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--encoder_params',
        type=str,
        default=ENCODER_PARAMS,
        help='JSON file with the encoder parameters.')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=None,
        help='Path to output wav file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=True,
        help='Use fast generation')
    parser.add_argument(
        '--wav_seed',
        type=str,
        default=None,
        help='The wav file to start generation from')
    parser.add_argument(
        '--gc_channels',
        type=int,
        default=None,
        help='Number of global condition embedding channels. Omit if no '
             'global conditioning.')
    parser.add_argument(
        '--gc_cardinality',
        type=int,
        default=None,
        help='Number of categories upon which we globally condition.')
    parser.add_argument(
        '--gc_id',
        type=int,
        default=None,
        help='ID of category to generate, if globally conditioned.')
#    parser.add_argument(
#        '--encoder_channels', type=int,
#        default=ENCODER_CHANNELS,
#        help='Number of channels in the text encoder net.')
#    parser.add_argument(
#        '--encoder_output_channels',
#        type=int,
#        default=ENCODER_OUTPUT_CHANNELS,
#        help='Number of output channels from the text encoder '
#             'net.')
#    parser.add_argument(
#        '--lc_channels',
#        type=int,
#        default=LOCAL_CONDITION_CHANNELS,
#        help='Number of channels in the upsampled local '
#             'condition fed to the audio wavenet.')
#    parser.add_argument(
#        '--encoder_layer_count',
#        type=int,
#        default=ENCODER_LAYER_COUNT,
#        help='Number of layers in the the text encoder.')
    parser.add_argument(
        '--text',
        type=str,
        default='Here we go.',
        help='Text to convert to speech.')

    arguments = parser.parse_args()
    if arguments.gc_channels is not None:
        if arguments.gc_cardinality is None:
            raise ValueError("Globally conditioning but gc_cardinality not "
                             "specified. Use --gc_cardinality=377 for full "
                             "VCTK corpus.")

        if arguments.gc_id is None:
            raise ValueError("Globally conditioning, but global condition was "
                              "not specified. Use --gc_id to specify global "
                              "condition.")

    return arguments


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def text_to_ascii(text):
    return [ord(char) for char in text]


def main():
    duration_ratio = 1.0
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)
    with open(args.encoder_params, 'r') as config_file:
        encoder_params = json.load(config_file)

    sess = tf.Session()

    net = WaveNetModel(
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'],
        histograms=False,
        global_condition_channels=args.gc_channels,
        global_condition_cardinality=args.gc_cardinality,
        local_condition_channels=encoder_params["local_condition_channels"])

    text_encoder = ConvNetModel(
        encoder_channels=encoder_params["encoder_channels"],
        histograms=False,
        output_channels=encoder_params['encoder_output_channels'],
        local_condition_channels=encoder_params['local_condition_channels'],
        upsample_rate=encoder_params['median_upsample_rate'],
        dilations=encoder_params['dilations'],
        gated_linear=False)

    text = args.text
    duration_in_characters = len(text)
    duration_in_samples = duration_in_characters * \
                          encoder_params['median_upsample_rate'] * \
                          duration_ratio
    duration_in_samples = int(duration_in_samples)

    samples = tf.placeholder(tf.int32)
    lc_placeholder = tf.placeholder(dtype=tf.float32)
    # ascii_placeholder = tf.placeholder(dtype=tf.int32)

    # Reshape the text to N characters x 1.
    ascii = [ord(achar) for achar in text]
    ascii = np.reshape(ascii, [-1])

    local_conditions = text_encoder.upsample(ascii, duration_in_samples)
    next_sample = net.predict_proba(samples, args.gc_id, lc_placeholder)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.all_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    quantization_channels = wavenet_params['quantization_channels']
    decode = mu_law_decode(samples, quantization_channels)
    waveform = np.random.randint(quantization_channels, size=(1,)).tolist()

    # First generate the local conditions from the text.
#    lc = sess.run(local_conditions,
#                  feed_dict={ascii_placeholder: ascii})
    lc = sess.run(local_conditions)

    last_sample_timestamp = datetime.now()
    for step in range(1,duration_in_samples):
#        if len(waveform) > args.window:
#            window = waveform[-args.window:]
#        else:
#            window = waveform
        receptive_field = net.receptive_field()
        if step >= receptive_field:
            window = waveform[step-receptive_field:step]
            lc_window = lc[:,step-receptive_field:step,:]
        else:
            window = waveform[:step]
            lc_window = lc[:,:step,:]

        outputs = next_sample

        # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs,
                              feed_dict={samples: window,
                                         lc_placeholder: lc_window})

        # Scale prediction distribution using temperature.
        np.seterr(divide='ignore')
        scaled_prediction = np.log(prediction) / args.temperature
        scaled_prediction = scaled_prediction - np.logaddexp.reduce(scaled_prediction)
        scaled_prediction = np.exp(scaled_prediction)
        np.seterr(divide='warn')

        # Prediction distribution at temperature=1.0 should be unchanged after scaling.
        if args.temperature == 1.0:
            np.testing.assert_allclose(prediction, scaled_prediction, atol=1e-5, err_msg='Prediction scaling at temperature=1.0 is not working as intended.')

        sample = np.random.choice(
            np.arange(net.softmax_channels), p=scaled_prediction)
        waveform.append(sample)

        # Show progress only once per second.
        current_sample_timestamp = datetime.now()
        time_since_print = current_sample_timestamp - last_sample_timestamp
        if time_since_print.total_seconds() > 1.:
            print('Sample {:3<d}/{:3<d}'.format(step + 1, duration_in_samples),
                  end='\r')
            last_sample_timestamp = current_sample_timestamp

        # If we have partial writing, save the result so far.
        if (args.wav_out_path and args.save_every and
                (step + 1) % args.save_every == 0):
            out = sess.run(decode, feed_dict={samples: waveform})
            # Keep only the non-blank generated samples.
            # The blank entries are the ones with value > 1.0
            non_blank = out <= 1.0
            out = out[non_blank]
            write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)

    # Introduce a newline to clear the carriage return from the progress.
    print()

    # Save the result as an audio summary.
    datestring = str(datetime.now()).replace(' ', 'T')
    writer = tf.train.SummaryWriter(logdir)
    tf.audio_summary('generated', decode, wavenet_params['sample_rate'])
    summaries = tf.merge_all_summaries()
    summary_out = sess.run(summaries,
                           feed_dict={samples: np.reshape(waveform, [-1, 1])})
    writer.add_summary(summary_out)

    # Save the result as a wav file.
    if args.wav_out_path:
        out = sess.run(decode, feed_dict={samples: waveform})
        write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)

    print('Finished generating. The result can be viewed in TensorBoard.')


if __name__ == '__main__':
    main()
