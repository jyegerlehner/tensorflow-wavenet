#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import fnmatch
import argparse
import librosa
import numpy as np
import os
import os.path

CORPUS_DIR = '/media/mass1/audio_data/VCTK-Corpus-stripped'
SAMPLE_RATE = 16000

''' Computes the number of samples and the number of text characters
    corresponding to each audio sample in the corpus.'''
def get_arguments():
    parser = argparse.ArgumentParser(description='Strip silence from audio')
    parser.add_argument('--corpus_dir', type=str, default=CORPUS_DIR,
                        help='The directory containing the .wav files.')
    parser.add_argument('--sample_rate', type=int, default=SAMPLE_RATE,
                        help='Sample rate.')
    return parser.parse_args()


def find_files(source_directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    wave_files = dict()
    text_files = dict()
    for root, dirnames, filenames in os.walk(source_directory):
        for filename in fnmatch.filter(filenames, pattern):
            wav_file = os.path.join(root, filename)
            (base, _) = os.path.splitext(filename)
            wave_files[base] = wav_file

        for filename in fnmatch.filter(filenames, "*.txt"):
            txt_file = os.path.join(root, filename)
            (base, _) = os.path.splitext(filename)
            text_files[base] = txt_file

    return wave_files, text_files


def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def main():
    args = get_arguments()
    print(args)
    logfile = open('out.csv', 'w')
    logfile.write("filename,samples,chars\n")
    wave_files, text_files = find_files(args.corpus_dir)
    for file_base in wave_files.keys():
        target_file = wave_files[file_base]
        if os.path.exists(target_file):
            audio, _ = librosa.load(target_file, sr=args.sample_rate,
                                    mono=True)
            if file_base in text_files:
                text = open(text_files[file_base], 'r').read()
                print("file:{}, samples:{}, chars:{}".format(file_base,
                                                             len(audio),
                                                             len(text)))
                logfile.write('{},{},{}\n'.format(file_base, len(audio),
                                                  len(text)))
            else:
                print("file:{} is missing text file.".format(file_base))

    logfile.close()


if __name__ == '__main__':
    main()
