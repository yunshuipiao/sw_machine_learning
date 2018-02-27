#!/usr/bin/env python

# encoding: utf-8

"""
@author: swensun

@github:https://github.com/yunshuipiao

@software: python

@file: index.py

@desc: 朴素贝叶斯分类

@hint:
"""

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

# 获取数据
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1] #1表示侮辱性言论，0表示正常
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 对输入的词汇表构建词向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = np.zeros(len(vocabList)) #生成零向量的array
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1 #有单词，该位置填充1
        else:
            print("the word: %s is not in my Vocabulary" % word)
            # pass
    return returnVec  #返回0，1的向量

# 词袋模型
def bagofWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return vocabList #返回非负整数的词向量

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  #文档数目
    numWord = len(trainMatrix[0])  #词汇表数目
    print(numTrainDocs, numWord)
    pAbusive = sum(trainCategory) / len(trainCategory) #p1, 出现侮辱性评论的概率 [0, 1, 0, 1, 0, 1]
    p0Num = np.zeros(numWord)
    p1Num = np.zeros(numWord)

    p0Demon = 0
    p1Demon = 0

    for i in range(numTrainDocs):
        if trainCategory[i] == 0:
            p0Num += trainMatrix[i] #向量相加
            p0Demon += sum(trainMatrix[i]) #向量中1累加其和
        else:
            p1Num += trainMatrix[i]
            p1Demon += sum(trainMatrix[i])
    p0Vec = p0Num / p0Demon
    p1Vec = p1Num / p1Demon

    return p0Vec, p1Vec, pAbusive

def trainNB1(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  #文档数目
    numWord = len(trainMatrix[0])  #词汇表数目
    pAbusive = sum(trainCategory) / len(trainCategory) #p1, 出现侮辱性评论的概率
    p0Num = np.ones(numWord)  #修改为1
    p1Num = np.ones(numWord)

    p0Demon = 2 #修改为2
    p1Demon = 2

    for i in range(numTrainDocs):
        if trainCategory[i] == 0:
            p0Num += trainMatrix[i] #向量相加
            p0Demon += sum(trainMatrix[i]) #向量中1累加其和
        else:
            p1Num += trainMatrix[i]
            p1Demon += sum(trainMatrix[i])
    p0Vec = np.log(p0Num / p0Demon)  #求对数
    p1Vec = np.log(p1Num / p1Demon)

    return p0Vec, p1Vec, pAbusive

#运动分类器对文本进行分类
def classifyNB(vec2Classify, p0Vc,  p1Vc, pClass1):
    p1 = sum(vec2Classify * p1Vc) * pClass1
    p0 = sum(vec2Classify * p0Vc) * (1-pClass1)
    # p1 = sum(vec2Classify * p1Vc) + np.log(pClass1)
    # p0 = sum(vec2Classify * p0Vc) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

# 构造样本测试
def testingNB():
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0v, p1v, pAb = trainNB0(trainMat, listClasses)
    # print(p0v, p1v, pAb)
    testEntry = ['love']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print(testEntry, 'classified as', classifyNB(thisDoc, p0v, p1v, pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = (setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0v, p1v, pAb))

if __name__ == '__main__':
    testingNB()







