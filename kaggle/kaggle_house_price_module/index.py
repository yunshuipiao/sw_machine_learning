import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from  sklearn.model_selection import train_test_split

import seaborn as sns
from scipy.stats import norm
from scipy import stats

if __name__ == '__main__':
    sns.set()
    df_train = pd.read_csv('train.csv')
    # print(df_train.head())  #输出前五行 81列
    # print(df_train.columns) # 查看各个特征的具体名称
    # print(df_train.describe())  #了解基本信息，包括最大最小值，平均数，标准差，不同阶段的数值
    print(df_train["SalePrice"].describe()) #某一列的基本统计特征
    # sns.distplot(df_train["SalePrice"]) #直方图查看某一特征数据的具体分布，蓝色曲线是默认参数 kde=True 的拟合曲线特征
    # plt.show()