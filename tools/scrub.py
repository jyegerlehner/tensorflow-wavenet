#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import fnmatch
import json
import numpy as np
import os
import os.path

SOURCE_DIR = '/media/mass1/audio_data/VCTK-Corpus-stripped/txt'
TARGET_DIR = '/media/mass1/audio_data/VCTK-Corpus-stripped/txt_scrubbed'
SAMPLE_RATE = 16000

def get_arguments():
    parser = argparse.ArgumentParser(description='Scrub the text of the'
                                     'corpus.')
    parser.add_argument('--source_dir', type=str, default=SOURCE_DIR,
                        help='The directory containing the .wav files.')
    parser.add_argument('--target_dir', type=str, default=TARGET_DIR,
                        help='The directory containing the .wav files.')
    parser.add_argument('--sample_rate', type=int, default=SAMPLE_RATE,
                        help='Sample rate.')
    return parser.parse_args()

def find_files(source_directory, target_directory):
    '''Recursively finds all text files.'''
    source_files = dict()
    target_files = dict()
    for root, dirnames, filenames in os.walk(source_directory):
        for filename in fnmatch.filter(filenames, "*.txt"):
            source_file = os.path.join(root, filename)
            base, ext = os.path.splitext(filename)
            source_files[base] = source_file

            relpath = os.path.relpath(path=source_file, start=source_directory)
            this_target = os.path.join(target_directory,relpath)
            target_files[base] = this_target

    return source_files, target_files


def fix_paren(text):
    splits = text.split(')')
    text = ' '.join(splits)
    return text


def fix_stuff_after_period(text):
    splits = text.split('.')
    if len(splits) > 1:
        new_text = splits[0] + '.'
        return new_text
    else:
        return text


def main():
    args = get_arguments()
    print(args)
    source_files, target_files = find_files(args.source_dir,
                                            args.target_dir)
    for file_base in source_files.keys():
        text_file = open(source_files[file_base], 'r')
        text = text_file.read()
        text_file.close()

        text = fix_paren(text)
        text = fix_stuff_after_period(text)

        target_filename = target_files[file_base]
        if not os.path.exists(os.path.dirname(target_filename)):
            try:
                os.makedirs(os.path.dirname(target_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        target_file = open(target_filename, 'w')
        target_file.write(text)
        target_file.close()


if __name__ == '__main__':
    main()
