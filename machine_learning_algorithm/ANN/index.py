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
def logistics(x):
    return 1 / (1 + np.exp(-x))

#逻辑函数的微分
def logistic_derivatice(x):
    return logistics(x) * (1 - logistics(x))

#构建ANN
class NeuralNetwork:
    # 构造函数 layers表示有多少层，每层有多少个神经元。
    # acvitation 为使用的激活函数名称，有默认值tanh.
    def __init__(self, layers, activation = 'tanh'):
        if activation == 'logisitic':
            self.activation = logistics
            self.activation_detiv = logistic_derivatice
        else:
            self.activation = tanh
            self.activation_detiv = tanh_deriv
        self.weight = []
        #len(layers)-1的目的是 输出层不需要赋予权值
        for i in range(1, len(layers) - 1):
            # 对当前层与前一层之间的连线进行权重赋值， 在-0.25-0.25之间
            self.weight.append( (2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weight.append( (2 * np.random.random((layers[i] + 1, layers[i + 1] + 1)) - 1) * 0.25)


    def fit(self, X, y, learning_rate = 0.2, epochs = 10000):
        # X表示训练集， 通常模拟成一个二维矩阵，每一层代表一个样本的不同特征
        # 每一列代表不同的样本， y指的是classLabel， 表示输出的分类标记
        # epochs表示循环次数

        # 将X转为numpy2维数组，至少二维
        X = np.atleast_2d(X)

        temp = np.ones([X.shape[0], X.shape[1] + 1])

        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]  # a是从x中任意抽取的一行数据

            #正向更新
            for l in range(len(self.weight)): #循环遍历每一层
                a.append(self.activation(np.dot(a[l], self.weight[l])))

            error = y[i] - a[-1]
            deltas = [error * self.activation_detiv(a[-1])]

            #反向传播
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weight[l].T) * self.activation_detiv(a[l]))
            deltas.reverse()

            for i in range(len(self.weight)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i]) #delta存的是误差
                self.weight[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weight)):
            a = self.activation(np.dot(a, self.weight[l]))
        return a



if __name__ == '__main__':
    # 进行测试
    # nn = NeuralNetwork([2, 1], 'tanh')
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([0, 0, 1, 1])
    # nn.fit(X, y)
    # for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    #     print(nn.predict(i))
    i = np.random.random((3, 2))
    print(i * 2)