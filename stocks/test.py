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


if __name__ == '__main__':
    stock_data = ts.get_hist_data("002230")
    # stock_data = stock_data.as_matrix()
    print(stock_data.head(10))
    

