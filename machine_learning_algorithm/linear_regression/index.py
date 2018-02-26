#!/usr/bin/env python

# encoding: utf-8

"""
@author: swensun

@github:https://github.com/yunshuipiao

@software: python

@file: index.py

@desc:线性回归：梯度下降

@hint:
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def init_data():
    data = np.loadtxt('data.csv', delimiter=',')
    return data


def linear_regression():
    learning_rate = 0.01 #步长
    initial_b = 0
    initial_m = 0
    num_iter = 1000 #迭代次数

    data = init_data()
    [b, m] = optimizer_two(data, initial_b, initial_m, learning_rate, num_iter)
    plot_data(data,b,m)
    print(b, m)
    return b, m


def optimizer(data, initial_b, initial_m, learning_rate, num_iter):
    b = initial_b
    m = initial_m

    for i in range(num_iter):
        b, m = compute_gradient(b, m, data, learning_rate)
        if i % 100 == 0:
            print(i, computer_error(b, m, data)) # 损失函数，即误差
    return [b, m]


def compute_gradient(b_cur, m_cur, data, learning_rate):
    b_gradient = 0
    m_gradient = 0

    N = float(len(data))
    #
    # 偏导数， 梯度
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]

        b_gradient += -(2 / N) * (y - ((m_cur * x) + b_cur))
        m_gradient += -(2 / N) * x * (y - ((m_cur * x) + b_cur)) #偏导数

    new_b = b_cur - (learning_rate * b_gradient)
    new_m = m_cur - (learning_rate * m_gradient)
    return [new_b, new_m]

def optimizer_two(data, initial_b, initial_m, learning_rate, num_iter):
    b = initial_b
    m = initial_m

    while True:
        before = computer_error(b, m, data)
        b, m = compute_gradient(b, m, data, learning_rate)
        after = computer_error(b, m, data)
        if abs(after - before) < 0.0000001:
            break
    return [b, m]

def compute_gradient_two(b_cur, m_cur, data, learning_rate):
    b_gradient = 0
    m_gradient = 0

    N = float(len(data))

    delta = 0.0000001

    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        # 利用导数的定义来计算梯度
        b_gradient = (error(x, y, b_cur + delta, m_cur) - error(x, y, b_cur - delta, m_cur)) / (2*delta)
        m_gradient = (error(x, y, b_cur, m_cur + delta) - error(x, y, b_cur, m_cur - delta)) / (2*delta)

    b_gradient = b_gradient / N
    m_gradient = m_gradient / N
    #
    new_b = b_cur - (learning_rate * b_gradient)
    new_m = m_cur - (learning_rate * m_gradient)
    return [new_b, new_m]


def error(x, y, b, m):
    return (y - (m * x) - b) ** 2


def computer_error(b, m, data):
    totalError = 0
    x = data[:, 0]
    y = data[:, 1]
    totalError = (y - m * x - b) ** 2
    totalError = np.sum(totalError, axis=0)
    return totalError / len(data)


def plot_data(data,b,m):

    #plotting
    x = data[:,0]
    y = data[:,1]
    y_predict = m*x+b
    plt.plot(x,y,'o')
    plt.plot(x,y_predict,'k-')
    plt.show()



def scikit_learn():
    data = init_data()
    y = data[:, 1]
    x = data[:, 0]
    x = (x.reshape(-1, 1))
    linreg = LinearRegression()
    linreg.fit(x, y)
    print(linreg.coef_)
    print(linreg.intercept_)


if __name__ == '__main__':
    # linear_regression()
    scikit_learn()



