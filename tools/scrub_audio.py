#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import fnmatch
import json
import numpy as np
import os
import os.path
import pandas as pd

MIN_SAMPLES = 5000
#ratio of samples to chars
MIN_RATIO = 400
MAX_RATIO = 2000

df = pd.read_csv('out.csv')
mat = df.as_matrix()
df['ratio'] = mat[:,1]/mat[:,2]

print(df.describe())
print(df.sort(['samples'], ascending=True))
print(df.sort(['ratio'], ascending=True))

blacklist_files = []
for index, row in df.iterrows():
    filebase = row['filename']
    samples = row['samples']
    chars = row['chars']
    ratio = row['ratio']

    if samples < MIN_SAMPLES:
        blacklist_files.append(filebase)
    elif ratio < MIN_RATIO:
        blacklist_files.append(filebase)
    elif ratio > MAX_RATIO:
        blacklist_files.append(filebase)


with open('blacklist.json', 'w') as target_file:
    json.dump(blacklist_files, target_file)



