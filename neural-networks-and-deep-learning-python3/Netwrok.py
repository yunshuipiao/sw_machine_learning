#!/usr/bin/env python

# encoding: utf-8

"""
@author: swensun

@github:https://github.com/yunshuipiao

@software: python

@file: Netwrok.py

@desc:

@hint:
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]




if __name__ == '__main__':
    net = Network([2, 3, 1])
    print(net.weights)
