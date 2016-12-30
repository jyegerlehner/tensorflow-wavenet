import fnmatch
import os
import random
import re
import threading
import traceback

import librosa
import numpy as np
import tensorflow as tf

FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'


def ascii_to_array(ascii_list):
    array = np.array(ascii_list)
    array = np.reshape(array, (-1, 1))
    return array


def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def random_file(base_names):
    for _ in base_names:
        file_index = random.randint(0, (len(base_names) - 1))
        yield base_names[file_index]


def find_files(directory, pattern='*.wav', blacklist=None):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            base, ext = os.path.splitext(filename)
            if blacklist is None or base not in blacklist:
                files.append(os.path.join(root, filename))
    return files


def matches_test_pattern(test_reg_exp, filename):
    return test_reg_exp.match(filename) is not None


def find_audio_and_text(corpus_directory,
                        is_train_not_test,
                        test_pattern, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    wave_files = dict()
    text_files = dict()

    files = find_files(corpus_directory)
    # Files that match the test pattern are used in testing, all other
    # files are used in training.
    test_reg_exp = re.compile(test_pattern)
    new_files = []
    for file in files:
        if matches_test_pattern(test_reg_exp, file) != is_train_not_test:
            new_files.append(file)
    files = new_files

    for root, dirnames, filenames in os.walk(corpus_directory):
        for filename in fnmatch.filter(filenames, "*.wav"):
            wav_file = os.path.join(root, filename)
            (base, _) = os.path.splitext(filename)
            wave_files[base] = wav_file

        for filename in fnmatch.filter(filenames, "*.txt"):
            txt_file = os.path.join(root, filename)
            (base, _) = os.path.splitext(filename)
            text_files[base] = txt_file

    return wave_files, text_files


def load_generic_audio(directory,
                       sample_rate,
                       test_pattern,
                       is_train_not_test,
                       blacklist):
    '''Generator that yields audio waveforms from the directory.'''
    audio_dict, text_dict = find_audio_and_text(directory,
                                                is_train_not_test,
                                                test_pattern=test_pattern,
                                                pattern=FILE_PATTERN)

    id_reg_exp = re.compile(FILE_PATTERN)
    for base_name in random_file(audio_dict.keys()):
        not_in_blacklist = blacklist is None or base_name not in blacklist
        also_in_text_dict = base_name in text_dict
        if not_in_blacklist and also_in_text_dict:
            filename = audio_dict[base_name]
            ids = id_reg_exp.findall(filename)
            if ids is None:
                # The file name does not match the pattern containing ids, so
                # there is no id.
                category_id = None
            else:
                # The file name matches the pattern for containing ids.
                category_id = int(ids[0][0])
            # load the audio.
            audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
            audio = audio.reshape(-1, 1)

            # load the text
            filename = text_dict[base_name]
            with open(filename, 'r') as text_file:
                # Make it lower case, since the VCTK corpus is quite small
                # and we won't see enough instances of upper case.
                text = text_file.read().lower()
                ascii_list = [ord(achar) for achar in text]

            yield audio, filename, category_id, ascii_list


def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(FILE_PATTERN)
    for file in files:
        ids = id_reg_exp.findall(file)
        if ids is None:
            return True
    return False


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 corpus_dir,
                 coord,
                 sample_rate,
                 gc_enabled,
                 max_sample_size,
                 test_pattern=None,
                 silence_threshold=None,
                 queue_size=64,
                 blacklist=None):
        self.please_stop = False
        self.corpus_dir = corpus_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.max_sample_size = max_sample_size
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.threads = []
        self.test_pattern = test_pattern
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.text_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,1))

        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32', 'int32'],
                                         shapes=[(None, 1), (None,1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder,
                                          self.text_placeholder])
        self.blacklist = blacklist
        self.do_test = len(test_pattern) > 0 if test_pattern is not None \
            else False
        if self.do_test:
            self.test_sample_placeholder = tf.placeholder(dtype=tf.float32,
                                                          shape=None)
            self.test_queue = tf.PaddingFIFOQueue(queue_size,
                                                  ['float32', 'int32'],
                                                  shapes=[(None, 1), (None,1)])
            self.test_text_placeholder = tf.placeholder(dtype=tf.int32,
                                                        shape=(None,1))
            self.test_enqueue = self.test_queue.enqueue(
                [self.test_sample_placeholder, self.test_text_placeholder])

        if self.gc_enabled:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
                                                shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])
            if self.do_test:
                self.test_id_placeholder = tf.placeholder(dtype=tf.int32,
                                                          shape=())
                self.test_gc_queue = tf.PaddingFIFOQueue(queue_size,
                                                         ['int32'],
                                                         shapes=[()])
                self.test_gc_enqueue = self.test_gc_queue.enqueue(
                    [self.test_id_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(corpus_dir, blacklist=blacklist)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(corpus_dir))
        if self.gc_enabled and not_all_have_id(files):
            raise ValueError("Global conditioning is enabled, but file names "
                             "do not conform to pattern having id.")
        # Determine the number of mutually-exclusive categories we will
        # accomodate in our embedding table.
        if self.gc_enabled:
            _, self.gc_category_cardinality = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(
                  self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

    def dequeue(self):
        audio, text = self.queue.dequeue()
        return audio, text

    def dequeue_gc(self):
        return self.gc_queue.dequeue()

    def dequeue_test_audio(self):
        return self.test_queue.dequeue()

    def dequeue_test_gc_id(self):
        return self.test_gc_queue.dequeue()

    def thread_main(self, sess, is_train_not_test):
        stop = False

        # Whether to enqueue training or test data.
        if is_train_not_test:
            # Enqueue training data.
            enqueue_op = self.enqueue
            sample_placeholder = self.sample_placeholder
            if self.gc_enabled:
                gc_enqueue_op = self.gc_enqueue
                id_placeholder = self.id_placeholder
        else:
            # Enqueue test data.
            enqueue_op = self.test_enqueue
            sample_placeholder = self.test_sample_placeholder
            if self.gc_enabled:
                gc_enqueue_op = self.test_gc_enqueue
                id_placeholder = self.test_id_placeholder

        try:
            # Go through the dataset multiple times
            while not stop:
                iterator = load_generic_audio(self.corpus_dir,
                                              self.sample_rate,
                                              self.test_pattern,
                                              is_train_not_test,
                                              self.blacklist)
                for audio, filename, category_id, char_list in iterator:
                    if self.coord.should_stop():
                        stop = True
                        break

                    # Skip any that are larger than max, as a mechanism for
                    # limiting memory usage.
                    if len(audio) < self.max_sample_size:
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: audio,
                                            self.text_placeholder:
                                                  ascii_to_array(char_list)})
                        if self.gc_enabled:
                            sess.run(self.gc_enqueue,
                                     feed_dict={self.id_placeholder:
                                                category_id})
                    else:
                        print("Skipping {}: audio length={}".format(
                            filename, len(audio)))

        except:
            self.please_stop = True
            self.coord.request_stop()
            self.queue.close()
            self.gc_queue.close()
            self.test_queue.close()
            self.test_gc_queue.close()
            print("Audio reader:")
            traceback.print_exc()

    def _start_thread(self, sess, is_train_not_test):
        thread = threading.Thread(target=self.thread_main,
                                  args=(sess, is_train_not_test,))
        thread.daemon = True  # Thread will close when parent quits.
        thread.start()
        self.threads.append(thread)


    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            self._start_thread(sess, is_train_not_test=True)
            if self.do_test:
                self._start_thread(sess, is_train_not_test=False)
        return self.threads
