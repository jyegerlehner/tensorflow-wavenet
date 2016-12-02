#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv('out.csv')
mat = df.as_matrix()
df['ratio'] = mat[:,1]/mat[:,2]

print(df.describe())
print(df.sort(['samples'], ascending=True))
print(df.sort(['ratio'], ascending=True))
mat = df.as_matrix()

plt.scatter(mat[:,2], mat[:,1])
plt.show()
hist, bins = np.histogram(mat[:,3], bins=50)
center = (bins[:-1]+bins[1:])/2
plt.bar(center, hist, align='center')
plt.show()
