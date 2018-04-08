#!/usr/bin/env python

# encoding: utf-8

"""
@author: swensun

@github:https://github.com/yunshuipiao

@software: python

@file: index.py

@desc:

@hint:
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import sys
import warnings
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from  sklearn import svm

seed = 782
np.random.seed(seed)

# load data
labeled_images = pd.read_csv('./input/train.csv')
images = labeled_images.iloc[0: 1000, 1:]
labels = labeled_images.iloc[0:1000, :1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)

# view image
# i = 1
# img = train_images.iloc[i].as_matrix()
# img = np.reshape(img, (28, 28))
# plt.imshow(img, cmap="gray")
# plt.title(train_labels.iloc[i, 0])
# plt.show()

# examining the pixel value
test_images[test_images>0]=1
train_images[train_images>0]=1
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
score = clf.score(test_images, test_labels)

test_data = pd.read_csv('./input/test.csv')
test_data[test_data>0] = 1
results = clf.predict(test_data[0: 1000])

df = pd.DataFrame(results)
df.index += 1
df.index.name = "ImageId"
df.columns = ['label']
df.to_csv('result.csv', header=True)

