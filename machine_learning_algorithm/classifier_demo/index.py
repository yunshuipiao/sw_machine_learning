#!/usr/bin/env python

# encoding: utf-8

"""
@author: swensun

@github:https://github.com/yunshuipiao

@software: python

@file: index.py

@desc: sklearn常用分类方法

@hint:
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from  sklearn import ensemble
from sklearn import naive_bayes
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



# 基本分类方法， IRIS数据集

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def load_iris_data():
    iris = load_iris()
    # X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
    iris.data, iris.target = shuffle_in_unison(iris.data, iris.target)
    x_train, x_test = iris.data[:100], iris.data[100:]
    y_train, y_test = iris.target[:100].reshape(-1, 1), iris.target[100:].reshape(-1, 1)
    return x_train, y_train, x_test, y_test


clfs = {'svm': svm.SVC(),\
        'decision_tree':tree.DecisionTreeClassifier(),
        'naive_gaussian': naive_bayes.GaussianNB(), \
        'naive_mul':naive_bayes.MultinomialNB(),\
        'K_neighbor' : neighbors.KNeighborsClassifier(),\
        'bagging_knn' : BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5,max_features=0.5), \
        'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5,max_features=0.5),
        'random_forest' : RandomForestClassifier(n_estimators=50),\
        'adaboost':AdaBoostClassifier(n_estimators=50),\
        'gradient_boost' : GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=1, random_state=0)
        }

def try_different_method(clf):
    clf.fit(x_train,y_train.ravel())
    score = clf.score(x_test,y_test.ravel())
    print('the score is :', score)




if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_iris_data()
    for clf_key in clfs.keys():
        print('the classifier is :', clf_key)
        clf = clfs[clf_key]
        try_different_method(clf)
