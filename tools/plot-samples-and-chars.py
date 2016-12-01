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
print(df.sort([' samples'], ascending=True))
print(df.sort(['ratio'], ascending=True))

plt.scatter(mat[:,2], mat[:,1])
plt.show()
