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
import random

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

    def feedforward(self, a):
        """return the output of the network if "a" is input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        desc: 随机梯度下降
        :param training_data: list of tuples (x,y)
        :param epochs: 训练次数
        :param mini_batch_size: 随机的最小集合
        :param eta: learning rate： 学习速率
        :param test_data: 测试数据，有的话会评估算法，但会降低运行速度
        :return:
        """
        if test_data:
            n_test = len(test_data)
            n = len(training_data)
            for j in range(epochs):
                random.shuffle(training_data)
                mini_batches = [
                    training_data[k: k + mini_batch_size]
                    for k in range(0, n, mini_batch_size)
                ]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta)
                if test_data:
                    print("Epoch {0}: {1} / {2}".format(
                        j, self.evaluate(test_data), n_test))
                else:
                    print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        梯度下降更新weights和biases， 用到backpropagation反向传播。
        :param mini_batch:
        :param eta:
        :return:
        """



if __name__ == '__main__':
    net = Network([2, 3, 1])
    print(net.weights)
