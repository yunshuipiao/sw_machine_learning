#!/usr/bin/env python

# encoding: utf-8

"""
@author: swensun

@github:https://github.com/yunshuipiao

@software: python

@file: test.py

@desc: 股票的线性回归预测

@hint:
"""
import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

length = 30

from sklearn import ensemble, linear_model, tree, svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class Stock:
    def __init__(self, name, code):
        self.code = code
        self.name = name

stocks_code = [
    # Stock("kedaxunfei", "002230"),
    # Stock("hengshengdianzi", "600570"),
    # Stock("ziguangguoxin", "002049"),
    # Stock("longjigufen", "601012"),
    # # Stock("360     ", "601360"),
    # Stock("shangpinzhaipei", "300616"),
    # # Stock("腾讯控股", "H00700"),
    # Stock("dongfangwangli", "300367"),
    # Stock("yiligufen", "600887"),
    # Stock("yongyouwangluo", "600588"),
    # Stock("dongfangguoxin", "300166"),
    # Stock("zhaoshangyinhang", "600036"),
    # Stock("zhongguopinan", "601318"),
    # Stock("shengheziyuan", "600392"),
]

if __name__ == '__main__':
    for s in stocks_code:
        stock_data = ts.get_hist_data(s.code)
        # print(stock_data.head(5))
        # print(stock_data.head(10))
        stock_data = stock_data.head(length).as_matrix()
        p_change = stock_data[:, 6]
        p_change[:-1] = p_change[1:]
        high_change = (stock_data[:, 1] - stock_data[:, 0]) / stock_data[:, 0] * 100
        # print(high_change[0:5])
        low_change = (stock_data[:, 3] - stock_data[:, 0]) / stock_data[:, 0] * 100
        # print(low_change[0:5])

        x = np.arange(0, length)
        plt.plot(x, high_change, "ro-", label="high")
        plt.plot(x, low_change, "go-", label="low")
        plt.title(s.name)
        plt.legend()
        plt.show()
    # total_data = stock_data
    # total_data = np.column_stack((total_data, p_change))

    # reg = linear_model.LinearRegression()
    # x_train, x_test, y_train, y_test = train_test_split(total_data, high_change, test_size=0.2, random_state=0)
    # reg.fit(x_train, y_train)
    # score = reg.score(x_test, y_test)
    # print(score)
    # latest_data = total_data[0:10, :]
    # latest_data = latest_data.mean(axis=0)
    # latest_data = latest_data.reshape(1, -1)
    # result = reg.predict(latest_data)
    # print(result)
    # print(high_change[0:10])
