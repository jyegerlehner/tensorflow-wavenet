"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""

from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.client import timeline

from wavenet import WaveNetModel, AudioReader, optimizer_factory, ConvNetModel

DATA_DIRECTORY = './VCTK-Corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 50
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-3
WAVENET_PARAMS = './wavenet_params.json'
ENCODER_PARAMS = './encoder_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
MAX_SAMPLE_SIZE = 120000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9
BLACKLIST='./blacklist.json'
#ENCODER_CHANNELS = 12
#ENCODER_OUTPUT_CHANNELS = 64
#LOCAL_CONDITION_CHANNELS = 16


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]


    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--store_metadata', type=bool, default=False,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters.')
    parser.add_argument('--encoder_params', type=str, default=ENCODER_PARAMS,
                        help='JSON file with the encoder parameters.')
    parser.add_argument('--blacklist', type=str, default=BLACKLIST,
                        help='JSON file containing set of file names to be '
                             'ignored while training (sans file extension).')
    parser.add_argument('--max_sample_size', type=int, default=MAX_SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many '
                        'samples.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Disabled by default')
    parser.add_argument('--silence_threshold', type=float,
                        default=SILENCE_THRESHOLD,
                        help='Volume threshold below which to trim the start '
                        'and the end from the training set samples.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer. Ignored by the '
                        'adam optimizer.')
    parser.add_argument('--histograms', type=_str_to_bool, default=False,
                         help='Whether to store histogram summaries.')
    parser.add_argument('--gc_channels', type=int, default=None,
                        help='Number of global condition channels.')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }

def get_input_batch(gc_enabled, test_interval, reader):
    audio_batch, text_batch = reader.dequeue()
    gc_id_batch = None
    if gc_enabled:
        gc_id_batch = reader.dequeue_gc()

    test_audio_batch = None
    test_gc_id_batch = None
    test_text_batch = None
    if test_interval > 0:
        test_audio_batch, test_text_batch = reader.dequeue_test_audio()
        if gc_enabled:
            test_gc_id_batch = reader.dequeue_test_gc_id()
    return (audio_batch, text_batch, gc_id_batch, test_audio_batch,
            test_text_batch, test_gc_id_batch)

def compute_test_loss(sess, test_steps, test_loss):
    accumulator = 0.0
    for iter in range(test_steps):
        test_loss_value = sess.run(test_loss)
        accumulator += test_loss_value
    accumulator /= test_steps
    return accumulator

def main():
    args = get_arguments()
    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    logdir_root = directories['logdir_root']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)
    with open(args.encoder_params, 'r') as f:
        encoder_params = json.load(f)

    print("test_pattern:", wavenet_params["test_pattern"])
    with open(args.blacklist, 'r') as f:
        blacklist = set(json.load(f))

    # Create coordinator.
    coord = tf.train.Coordinator()
    test_interval = wavenet_params['test_interval']
    do_test = test_interval > 0
    # Load raw waveform from VCTK corpus.
    with tf.name_scope('create_inputs'):
        # Allow silence trimming to be skipped by specifying a threshold near
        # zero.
        silence_threshold = args.silence_threshold if args.silence_threshold > \
                                                      EPSILON else None
        gc_enabled = args.gc_channels is not None
        reader = AudioReader(
            args.data_dir,
            coord,
            sample_rate=wavenet_params['sample_rate'],
            gc_enabled=gc_enabled,
            max_sample_size=args.max_sample_size,
            test_pattern=wavenet_params['test_pattern'],
            silence_threshold=args.silence_threshold,
            blacklist=blacklist,
            do_test=do_test)

        (audio_batch, text_batch, gc_id_batch, test_audio_batch,
         test_text_batch, test_gc_id_batch) = \
            get_input_batch(gc_enabled, test_interval, reader)

    # Create text encoder network.
    text_encoder = ConvNetModel(
        encoder_channels=encoder_params["encoder_channels"],
        histograms=args.histograms,
        output_channels=encoder_params['encoder_output_channels'],
        local_condition_channels=encoder_params['local_condition_channels'],
        dilations=encoder_params['dilations'],
        gated_linear=False,
        density_conditioned=True)

    # Create network.
    net = WaveNetModel(
        dilations=wavenet_params["dilations"],
        filter_width=wavenet_params["filter_width"],
        residual_channels=wavenet_params["residual_channels"],
        dilation_channels=wavenet_params["dilation_channels"],
        skip_channels=wavenet_params["skip_channels"],
        quantization_channels=wavenet_params["quantization_channels"],
        use_biases=wavenet_params["use_biases"],
        histograms=args.histograms,
        global_condition_channels=args.gc_channels,
        global_condition_cardinality=reader.gc_category_cardinality,
        local_condition_channels=encoder_params["local_condition_channels"],
        ctc_loss=False,
        gated_linear=False)

    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None

    text_batch = tf.squeeze(text_batch)
    lc_batch = text_encoder.upsample(text_batch, tf.shape(audio_batch)[0])
    if test_text_batch is not None:
        test_text_batch = tf.squeeze(test_text_batch)
        test_lc_batch = text_encoder.upsample(test_text_batch,
                                          tf.shape(test_audio_batch)[0])
    loss = net.loss(input_batch=audio_batch,
                    global_condition_batch=gc_id_batch,
                    local_condition_batch=lc_batch,
                    l2_regularization_strength=args.l2_regularization_strength)

    if test_text_batch is not None:
        test_loss = net.loss(input_batch=test_audio_batch,
                         global_condition_batch=test_gc_id_batch,
                         local_condition_batch=test_lc_batch,
                         l2_regularization_strength=
                            args.l2_regularization_strength,
                         loss_prefix='test_')

    optimizer = optimizer_factory[args.optimizer](
                    learning_rate=args.learning_rate,
                    momentum=args.momentum)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up logging for TensorBoard.
    writer = tf.train.SummaryWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.merge_all_summaries()

    # Set up session
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=tf.trainable_variables())

        try:
            saved_global_step = load(saver, sess, restore_from)
            if is_overwritten_training or saved_global_step is None:
                # The first training step will be saved_global_step + 1,
                # therefore we put -1 here for new or overwritten trainings.
                saved_global_step = -1

        except:
            print("Something went wrong while restoring checkpoint. "
                  "We will terminate training to avoid accidentally overwriting "
                  "the previous model.")
            raise

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        threads.extend(reader.start_threads(sess))
        step = None
        test_loss_value = 0.0
        try:
            last_saved_step = saved_global_step
            for step in range(saved_global_step + 1, args.num_steps):
                if coord.should_stop():
                    break
                start_time = time.time()
                if args.store_metadata and step % 50 == 0:
                    # Slow run that stores extra information for debugging.
                    print('Storing metadata')
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    summary, loss_value, _ = sess.run(
                        [summaries, loss, optim],
                        options=run_options,
                        run_metadata=run_metadata)
                    writer.add_summary(summary, step)
                    writer.add_run_metadata(run_metadata,
                                            'step_{:04d}'.format(step))
                    tl = timeline.Timeline(run_metadata.step_stats)
                    timeline_path = os.path.join(logdir, 'timeline.trace')
                    with open(timeline_path, 'w') as f:
                        f.write(tl.generate_chrome_trace_format(show_memory=True))
                else:
                    summary, loss_value, _ = sess.run([summaries, loss, optim])
                    writer.add_summary(summary, step)

                # Print an asterisk only if we've recomputed test loss.
                test_computed = ' '
                if test_interval > 0 and step % test_interval == 0:
                    test_steps = wavenet_params["test_steps"]
                    test_loss_value = compute_test_loss(sess, test_steps,
                                                        test_loss)
                    test_computed = '*'


                duration = time.time() - start_time
                print('step {:d} - loss = {:.3f}, last test loss = {:3f},'
                      ' ({:.3f} sec/step) {}'
                      .format(step, loss_value, test_loss_value, duration,
                              test_computed))

                if step % args.checkpoint_every == 0:
                    save(saver, sess, logdir, step)
                    last_saved_step = step

        except Exception, e:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
            print()
            coord.request_stop(e)
        finally:
            if step > last_saved_step:
                save(saver, sess, logdir, step)
            coord.request_stop()
            #coord.join(threads)


if __name__ == '__main__':
    main()
