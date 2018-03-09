#!/usr/bin/env python

# encoding: utf-8

"""
@author: swensun

@github:https://github.com/yunshuipiao

@software: python

@file: index.py

@desc: sklearn中常用的回归方法和集成方法

@hint:
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from  sklearn import ensemble



def f(x1, x2):
    y = 0.5 * np.sin(x1) + 0.5 * np.cos(x2) + 0.1 * x1 + 3
    return y


def load_data():
    x1_train = np.linspace(0,  50, 500)
    x2_train = np.linspace(-10, 10, 500)

    # np.random.random()：0-1的随机浮点数
    data_train = np.array([[x1, x2, f(x1, x2) + (np.random.random() - 0.5)] for x1, x2 in zip(x1_train, x2_train)])

    x1_test = np.linspace(0, 50, 100) + 0.5 * np.random.random(100)
    x2_test = np.linspace(-10, 10, 100) + 0.02 * np.random.random(100)
    data_test = np.array([[x1, x2, f(x1, x2)] for x1, x2 in zip(x1_test, x2_test)])

    return data_train, data_test

def try_different_method(clf):
    data_train, data_test = load_data()
    x_train, y_train = data_train[:, :2], data_train[:, 2]
    x_test, y_test = data_test[:, :2], data_test[:, 2]

    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    result = clf.predict(x_test)

    plt.figure()
    plt.plot(np.arange(len(result)), y_test, 'go-', label="true value")
    plt.plot(np.arange(len(result)), result, 'ro-', label="prefict value")
    plt.title('score: %f' % score)
    plt.legend()
    plt.show()


def linear_regression():
    linear_reg = linear_model.LinearRegression()
    try_different_method(linear_reg)

def tree_regression():
    tree_reg = tree.DecisionTreeRegressor()
    try_different_method(tree_reg)

def svm_regression():
    svm_reg = svm.SVR()
    try_different_method(svm_reg)

def knn_regression():
    knn_reg = neighbors.KNeighborsRegressor()
    try_different_method(knn_reg)

def random_forest_regression():
    rf_reg = ensemble.RandomForestRegressor(n_estimators=20)
    try_different_method(rf_reg)

def adaboost_regression():
    adaboost_reg = ensemble.AdaBoostRegressor()
    try_different_method(adaboost_reg)

def GBRT_regression():
    gbrt_reg = ensemble.GradientBoostingRegressor(n_estimators=100)
    try_different_method(gbrt_reg)



if __name__ == '__main__':
    linear_regression()
    tree_regression()
    svm_regression()
    knn_regression()
    random_forest_regression()






