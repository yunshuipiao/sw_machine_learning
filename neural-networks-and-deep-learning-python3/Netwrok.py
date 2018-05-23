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

import mnist_loader


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


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
            test_data = list(test_data)
            n_test = len(test_data)
        training_data = list(training_data)
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
                print("Epoch {}: {} / {}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def evaluate(self, test_data):
        """
        评估测试集的准确性
        :param test_data:
        :return:
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def update_mini_batch(self, mini_batch, eta):
        """
        梯度下降更新weights和biases， 用到backpropagation反向传播。
        :param mini_batch:
        :param eta:
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        :param x:
        :param y:
        :return: (nabla_b, nabla_w): gradient for 损失函数，类似于biaes， weight。
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # 存储所有的激活值
        zs = []  # 存储所有的z向量
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """
        :param output_activations:
        :param y:
        :return: 给定输出激发， 返回偏导数向量
        """
        return (output_activations - y)


if __name__ == '__main__':
    training_data, validation_data, test_data = \
        mnist_loader.load_data_wrapper()
    # net = Network([784, 30, 10])
    # net.SGD(training_data, 30, 10, 3.0, test_data)
