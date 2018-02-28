#!/usr/bin/env python

# encoding: utf-8

"""
@author: swensun

@github:https://github.com/yunshuipiao

@software: python

@file: dating_index.py

@desc: kNN用于约会网站

@hint:
"""

import numpy as np
import matplotlib.pyplot as plt

def classify(inX, dataSet, labels, k):
    """
    定义knn算法分类器函数
    :param inX: 测试数据
    :param dataSet: 训练数据
    :param labels: 分类类别
    :param k: k值
    :return: 所属分类
    """

    dataSetSize = dataSet.shape[0]  #shape（m, n）m列n个特征
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5  #欧式距离
    sortedDistIndicies = distances.argsort()  #排序并返回index

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]][0]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #default 0

    sortedClassCount = sorted(classCount.items(), key=lambda d:d[1], reverse=True)
    return sortedClassCount[0][0]

# 文本转换成numpy
def file2matrix(filepath="datingSet.csv"):
    dataSet = np.loadtxt(filepath)
    returnMat = dataSet[:, 0:-1]
    classlabelVector = dataSet[:, -1:]
    return returnMat, classlabelVector

#简单分析数据

# 2， 3列
def show_2_3_fig():
    data, cls = file2matrix()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 1], data[: ,2], c=cls)
    plt.xlabel("playing game")
    plt.ylabel("Icm Cream")
    plt.show()

# 2， 3列
def show_1_3_fig():
    data, cls = file2matrix()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 2],c=cls)
    plt.xlabel("Mile of plane")
    plt.ylabel("Ice Cream")
    plt.show()

# 2， 3列
def show_1_2_fig():
    data, cls = file2matrix()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], c=cls)
    plt.xlabel("Mile of plane")
    plt.ylabel("playing game")
    plt.show()

# 数据归一化
def autoNorm(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal

    normDataSet = np.zeros(dataSet.shape)
    m, n = dataSet.shape  # 行， 特征
    normDataSet = dataSet - minVal
    normDataSet = normDataSet / ranges
    return normDataSet, ranges, minVal

# 定义测试算法的函数
def datingClassTest(h=0.1):
    hoRatio = h
    datingDataMat, datingLabels = file2matrix()
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m, n = normMat.shape
    numTestVecs = int(m * hoRatio)  #测试数据行数
    errorCount = 0  # 错误分类数


    # 用前10%的数据做测试
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m],  3)
        # print('the classifier came back with: %d,the real answer is: %d' % (int(classifierResult), int(datingLabels[i])))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))

# 简单进行预测
def classifypersion():
    resultList = ["none", 'not at all','in small doses','in large doses']
    # 模拟数据
    ffmiles = 15360
    playing_game = 8.545204
    ice_name = 1.340429

    datingDataMat, datingLabels = file2matrix()
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffmiles, playing_game, ice_name])
    # 预测数据归一化
    inArr = (inArr - minVals) / ranges
    classifierResult = classify(inArr, normMat, datingLabels, 3)
    print(resultList[int(classifierResult)])



if __name__ == '__main__':
    # data, cls = file2matrix()
    # autoNorm(data)
    # datingClassTest()
    classifypersion()
    # show_1_2_fig()
