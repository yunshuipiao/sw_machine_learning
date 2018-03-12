#!/usr/bin/env python

# encoding: utf-8

"""
@author: swensun

@github:https://github.com/yunshuipiao

@software: python

@file: index.py

@desc: ANN的实现

@hint:
"""

import numpy as np

#双曲函数
def tanh(x):
    return np.tanh(x)

#双曲函数的微分
def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)

#逻辑函数
def logistic(x):
    return 1 / (1 + np.exp(-x))

#逻辑函数的微分
def logistic_derivatice(x):
    return logistic(x) * (1 - logistic(x))

#构建ANN
class NeuralNetwork:
    pass