#!/usr/bin/env python

# encoding: utf-8

"""
@author: swensun

@github:https://github.com/yunshuipiao

@software: python

@file: index_two.py

@desc:

@hint:   https://juejin.im/post/5a87a7026fb9a063475f8706  参考
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def show_sigmoid():
    x = [1, 2, 3, 4, 6, 7, 8, 9, 10]
    y = [0, 0, 0, 0, 1, 1, 1, 1, 1]

    train_X = np.asarray(x)
    train_Y = np.asarray(y)

    fig = plt.figure()
    plt.xlim(-1, 12)
    plt.ylim(-0.5, 1.5)
    plt.scatter(train_X, train_Y)

    s_X = np.linspace(-2, 12, 100)
    s_Y = 1 / (1 + np.exp(-6 * (s_X - 5)))
    plt.plot(s_X, s_Y)
    plt.show()

def init_data():
    x = [1, 2, 3, 4, 6, 7, 8, 9, 10]
    y = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    train_X = np.asarray(np.row_stack((np.ones(shape=(1, len(x))), x)), dtype=np.float64)
    train_Y = np.asarray(y, dtype=np.float64)
    train_W = np.asarray([-1, 1], dtype=np.float64).reshape(1, 2)
    return train_X, train_Y, train_W
#
#
def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def lossfunc(X, Y, W):
    n = len(Y)
    return (-1 / n) * np.sum(Y * np.log(sigmoid(np.matmul(W, X))) + (1 - Y) * np.log((1 - sigmoid(np.matmul(W, X)))))


def gradient_descent(X, Y, W, learningrate=0.001, trainingtimes=500):
    n = len(Y)
    for i in range(trainingtimes):
        W = W - (learningrate / n) * np.sum((sigmoid(np.matmul(W, X)) - Y) * X, axis=1)

        #for gif
        if 0 == i % 1000 or (100 > i and 0 == i % 2):
            b_Trace.append(W[0, 0])
            w_Trace.append(W[0, 1])
            loss_Trace.append(lossfunc(X, Y, W))
    return W
#
def update(i):
    try:
        ax.lines.pop(0)
    except Exception:
        pass
    plot_X = np.linspace(-1, 12, 100)
    W = np.asarray([b_Trace[i], w_Trace[i]]).reshape(1, 2)
    X = np.row_stack((np.ones(shape=(1, len(plot_X))), plot_X))
    plot_Y = sigmoid(np.matmul(W, X))
    line = ax.plot(plot_X, plot_Y[0], 'r-', lw=1)
    ax.set_xlabel(r"$Cost\ %.6s$" % loss_Trace[i])
    return line
#
#
if __name__ == '__main__':

    # init data
    loss_Trace = []
    w_Trace = []
    b_Trace = []

    x = [1, 2, 3, 4, 6, 7, 8, 9, 10]
    y = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    train_X = np.asarray(np.row_stack((np.ones(shape=(1, len(x))), x)), dtype=np.float64)
    train_Y = np.asarray(y, dtype=np.float64)
    train_W = np.asarray([-1, 1], dtype=np.float64).reshape(1, 2)
    final_W = gradient_descent(train_X, train_Y, train_W, 0.3, 100000)

    print(final_W)
    print(np.asarray([b_Trace, w_Trace]))
    print(loss_Trace)

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    ani = animation.FuncAnimation(fig, update, frames=len(w_Trace), interval=100)

    plt.show()








