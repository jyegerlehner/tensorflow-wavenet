#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import fnmatch
import argparse
import csv
import librosa
import math
import numpy as np
import os
import os.path

# File produced by chars-to-samples.py
STATS_FILE = 'out.csv'
DENSITY_QUANT_LEVELS = 50
MIN_SAMPLE_DENSITY = 500.0
MAX_SAMPLE_DENSITY = 2000.0
DENSITY_SPAN = MAX_SAMPLE_DENSITY - MIN_SAMPLE_DENSITY


''' Computes the number of samples and the number of text characters
    corresponding to each audio sample in the corpus.'''
def get_arguments():
    parser = argparse.ArgumentParser(description='Compute sample densities.')
    parser.add_argument('--stats_file', type=str, default=STATS_FILE,
                        help='File containing numbers of chars and saples.')
    return parser.parse_args()

def quantize(density_val):
    scaled = None
    if (density_val < MIN_SAMPLE_DENSITY):
        print("Density < MIN:{}".format(density_val))
    elif (density_val > MAX_SAMPLE_DENSITY):
        print("Density > MAX:{}".format(density_val))
    else:
        scaled = (density_val - MIN_SAMPLE_DENSITY) / DENSITY_SPAN
        scaled = math.floor(scaled*DENSITY_QUANT_LEVELS)
    return scaled

def unquantize(level):
    ratio = float(level) / float( DENSITY_QUANT_LEVELS)
    return math.floor(ratio*DENSITY_SPAN + MIN_SAMPLE_DENSITY)


def main():
#    for level in range(DENSITY_QUANT_LEVELS):
#        val = unquantize(level)
#        roundtrip = quantize(val)
#        print("Original: {}, Roundtripped: {}".format(level, roundtrip))

#    for x in np.arange(MIN_SAMPLE_DENSITY, MAX_SAMPLE_DENSITY):
#        level = quantize(x)
#        roundtrip = unquantize(level)
#        print("Original val: {}, roundtrip: {}".format(x, roundtrip))

    args = get_arguments()
    counts = {}
    with open(args.stats_file, 'r') as stats_file:
        reader = csv.reader(stats_file, delimiter=',')
        got_first = False
        for row in reader:
            if got_first:
                samples = float(row[1])
                chars = float(row[2])

                density = samples/chars
                quantized = quantize(density)
                if quantized in counts:
                    counts[quantized] += 1
                else:
                    counts[quantized] = 1
            else:
                got_first = True

        for level in range(DENSITY_QUANT_LEVELS):
            if level in counts:
                print("level:{} counts:{}".format(level, counts[level]))
            else:
                print("No counts for level {}.".format(level))

if __name__ == '__main__':
    main()
