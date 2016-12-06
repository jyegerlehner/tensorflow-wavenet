import fnmatch
import os
import random
import re
import threading

import librosa
import numpy as np
import tensorflow as tf

FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'


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


def find_audio_and_text(corpus_directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    wave_files = dict()
    text_files = dict()
    for root, dirnames, filenames in os.walk(corpus_directory):
        for filename in fnmatch.filter(filenames, pattern):
            wav_file = os.path.join(root, filename)
            (base, _) = os.path.splitext(filename)
            wave_files[base] = wav_file

        for filename in fnmatch.filter(filenames, "*.txt"):
            txt_file = os.path.join(root, filename)
            (base, _) = os.path.splitext(filename)
            text_files[base] = txt_file

    return wave_files, text_files


def load_generic_audio(directory, sample_rate, blacklist):
    '''Generator that yields audio waveforms from the directory.'''
    audio_dict, text_dict = find_audio_and_text(directory)

    blacklist=blacklist

    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    #randomized_files = randomize_files(files)
    for base_name in random_files(audio_dict.keys()):
        not_in_blacklist = blacklist is None or base_name not in blacklist
        also_in_text_dict = base_name in text_dict
        if not_in_blacklist and also_in_text_dict:
            ids = id_reg_exp.findall(filename)
            if ids is None:
                # The file name does not match the pattern containing ids, so
                # there is no id.
                category_id = None
            else:
                # The file name matches the pattern for containing ids.
                category_id = int(ids[0][0])
            # load the audio.
            filename = audio_dict[base_name]
            audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
            audio = audio.reshape(-1, 1)

            # load the text
            filename = text_dict[base_name]
            with open(filename, 'r') as text_file:
                # Make it lower case, since the VCTK corpus is quite small
                # and we won't see enough instances of upper case.
                text = text_file.read().lower()
                character_list = [ ord(achar) for char in text]

            yield audio, filename, category_id, character_list


def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


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
                 audio_dir,
                 coord,
                 sample_rate,
                 gc_enabled,
                 sample_size,
                 silence_threshold=None,
                 queue_size=32,
                 blacklist=None):
        self.corpus_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.text_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32', 'int32'],
                                         shapes=[(None, 1), (None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder,
                                           self.text_placeholder])
        self.blacklist = blacklist

        if self.gc_enabled:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
                                                shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir, blacklist=blacklist)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))
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

    def dequeue(self, num_elements):
        audio, text = self.queue.dequeue_many(num_elements)
        return audio, text

    def dequeue_gc(self, num_elements):
        return self.gc_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        buffer_ = np.array([])
        stop = False
        try:
            # Go through the dataset multiple times
            while not stop:
                iterator = load_generic_audio(self.corpus_dir,
                                              self.sample_rate,
                                              self.blacklist)
                for audio, filename, category_id, char_list in iterator:
                    if self.coord.should_stop():
                        stop = True
                        break

                    # Cut samples into fixed size pieces
                    buffer_ = np.append(buffer_, audio)
                    while len(buffer_) > 0:
                        piece = np.reshape(buffer_[:self.sample_size], [-1, 1])
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece,
                                            self.text_placeholder: char_list})
                        buffer_ = buffer_[self.sample_size:]
                        if self.gc_enabled:
                            sess.run(self.gc_enqueue,
                                     feed_dict={self.id_placeholder:
                                                category_id})
        except Exception, e:
            # Report exceptions to the coordinator.
            self.coord.request_stop(e)
        finally:
            # Terminate as usual.  It is innocuous to request stop twice.
            self.coord.request_stop()
            self.coord.join(threads)


    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            #thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
