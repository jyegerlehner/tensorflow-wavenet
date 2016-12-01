#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import fnmatch
import argparse
import librosa
import numpy as np
import os

SOURCE_DIR = '/media/mass1/audio_data/VCTK-Corpus'
TARGET_DIR = '/media/mass1/audio_data/VCTK-Corpus-stripped'
SAMPLE_RATE = 16000
THRESHOLD = 0.3


def get_arguments():
    parser = argparse.ArgumentParser(description='Strip silence from audio')
    parser.add_argument('--source', type=str, default=SOURCE_DIR,
                        help='The directory containing the .wav files.')
    parser.add_argument('--target', type=str, default=TARGET_DIR,
                        help='The directory to place the stripped'
                        ' .wav files.')
    parser.add_argument('--sample_rate', type=int, default=SAMPLE_RATE,
                        help='Sample rate.')
    parser.add_argument('--threshold', type=float, default=THRESHOLD,
                        help='Silence trimming threshold.')
    return parser.parse_args()

def find_files(source_directory, target_directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    source_files = []
    target_files = []
    for root, dirnames, filenames in os.walk(source_directory):
        for filename in fnmatch.filter(filenames, pattern):
            this_source = os.path.join(root, filename)
            source_files.append(this_source)
            relpath = os.path.relpath(path=this_source, start=source_directory)
            this_target = os.path.join(target_directory,relpath)
            target_files.append(this_target)
    return source_files, target_files

def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio, hop_length=512)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]
    start_index = indices[0] if indices.size else 0
    si1 = start_index
    SILENCE_BUFFER = 2000
    start_index -= SILENCE_BUFFER
    if start_index < 0:
        start_index = 0
    end_index = indices[-1] if indices.size else 0
    ei1 = end_index
    end_index += SILENCE_BUFFER
    if end_index >= len(audio):
        end_index = len(audio)-1

    print("ei:{} si:{} start_index:{} end_index:{} len(audio):{}".format(ei1, si1, start_index, end_index, len(audio)))

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[start_index:end_index]

def main():
    args = get_arguments()
    print(args)
    source_files, target_files = find_files(args.source, args.target)
    for (source_file, target_file) in zip(source_files, target_files):
        audio, _ = librosa.load(source_file, sr=args.sample_rate, mono=True)
        trimmed = trim_silence(audio, args.threshold)
        print("original len:{}, trimmed len:{}".format(len(audio),len(trimmed)))
        if not os.path.exists(os.path.dirname(target_file)):
            try:
                os.makedirs(os.path.dirname(target_file))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        if len(trimmed) > 0:
            librosa.output.write_wav(target_file, trimmed,
                                     args.sample_rate)




if __name__ == '__main__':
    main()
