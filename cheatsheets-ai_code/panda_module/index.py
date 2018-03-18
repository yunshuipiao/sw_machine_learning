
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _sum(x):  #定义一个函数
    print(type(x))
    return x.sum()

if __name__ == '__main__':
    #基本序列 Series
    s_data = pd.Series([1, 3, 5, 7, np.NaN, 9, 11])
    dates = pd.date_range('20170220', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD')) #randn:随机生成，满足标准正态分布
    # print(df)

    #DataFrame
    # datas = pd.date_range("20180103", periods=6)
    # data = pd.DataFrame(np.random.randn(6, 4), index=datas, columns=list("ABCD"))
    # print(data)
    # print(data.shape)
    # print(data.values)

    #基本操作1：
    # d_data = {'A': 1, 'B': pd.Timestamp('20180103'), 'C':range(4), "D": np.arange(4)}
    # print(d_data)
    # df_data = pd.DataFrame(d_data)
    # print(df_data)
    # print(df_data.dtypes)
    # print(df_data.A)
    # print(df_data.B)
    # print(type(df_data.B))


    #基本操作2：
    # datas = pd.date_range('20180103', periods=6)
    # data = pd.DataFrame(np.random.randn(6, 4), index=datas, columns=list('ABCD'))
    # print(data)
    # print(data.head()) #默认前五行
    # print(data.head(1))
    # print(data.tail())
    # print(data.tail(1))
    # print(data.index)  #行索引
    # print(data.columns) #列
    # print(data.values) #数据值
    # print(data.describe()) #详细信息

    #基本操作3：
    # dates = pd.date_range('20170220', periods=6)
    # data = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    # print(data)
    # print(data.T) #转置
    # print(data.shape)
    # print(data.T.shape)
    # print(data.sort_index(axis=1)) #列索引排序
    # print(data.sort_index(axis=1, ascending=False)) #列索引降序排列
    # print(data.sort_index(axis=0, ascending=False)) #行索引降序排列
    # print(data.sort_values(by='A')) #A列值升序排列

    #基本操作4：
    # dates = pd.date_range('20170220', periods=6)
    # data = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    # print(data)
    # print(data.A)
    # print(data["A"]) #输出A列
    # print(data[2:4])  #输出3，4 行， 0开始
    # print(data['20170222': '20170223']) #输出对应行
    # print(data.loc['20170222': '20170223']) #对应行
    # print(data.iloc[2:4]) #输出3， 4行
    # print(data.loc[:, ['B', 'C']]) #输出B， C两列

    #基本操作5：
    # dates = pd.date_range('20170220', periods=6)
    # data = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    # print(data)
    # print(data[data.A > 0])  #输出A列中大于0的行
    # print(data[data > 0])  #输出大于0的数据，小于0的用NAN补位
    # data2 = data.copy()
    # print(data2)
    # tag = ['a'] * 2 + ['b'] * 2 + ['c'] * 2
    # data2['TAG'] = tag  #data2中增加TAG列并赋值
    # print(data2)
    # print(data2[data2.TAG.isin(['a', 'c'])]) #打印TAG列中a，c的行

    #基本操作6：
    # dates = pd.date_range('20170220', periods=6)
    # data = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    # print(data)
    # data.iat[0, 0] = 100 #修改元素[0, 0]
    # print(data)
    # data.A = range(6)
    # print(data)
    # data.B = 200
    # print(data)
    # data.iloc[:, 2: 5] = 1000  #3,4列元素赋值1000
    # print(data)

    #基本操作7：
    # dates = pd.date_range('20170220', periods=6)
    # df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    # print(df)
    # dfl = df.reindex(index = dates[0:4], columns = list(df.columns) + ['E']) #重定义索引，并添加E行
    # print(dfl)
    # dfl.loc[dates[1:3], ['E']] = 2 #E列中的2，3 行赋值为2
    # print(dfl)
    #
    # print(dfl.dropna())  #去掉存在NAN的行
    #
    # print(dfl.fillna(5)) #替换NAN为5
    # print(pd.isnull(dfl)) #判断每个元素是否为null
    #
    # print(dfl.mean()) #求列的平均值
    # print(dfl.cumsum()) #对每列累加

    # print(dfl.mean(axis=1)) #对行求平均值
    # s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2) #生成序列并向右平移2位
    # print(s)
    #
    # print(df.sub(s, axis='index')) #df与s做减法运算 df - s
    # print(df.cumsum()) # 对每列进行累加计算
    # print(df.apply(lambda x: x.max() - x.min())) #每列的最大值减去最小值

    #基本操作9：
    # print(df.apply(_sum))  #apply可以指定一个函数作为参数
    # s = pd.Series(np.random.randint(10, 20, size=15))
    # print(s)
    # print(s.value_counts()) #统计序列中每个元素出现的次数
    # print(s.mode())  #返回出现次数最多的元素

    #基本操作10：
    # df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))
    # print(df)
    # dfl = pd.concat([df.iloc[:3], df.iloc[3:7], df.iloc[7:]])  #合并函数
    # print(dfl)
    # print(df == dfl) #判断两个DataFrame的元素是否相等

    #基本操作11：
    # print(df)
    # left = pd.DataFrame({'key':['foo', 'foo'], 'lval': [1, 2]})
    # right = pd.DataFrame({'key':['foo', 'foo'], 'rval': [4, 5]})
    # print(left)
    # print(right)
    # print(pd.merge(left, right, on='key')) #通过key合并数据
    #
    # s = pd.Series(np.random.randint(1, 5, size=4), index=list('ABCD'))
    # print(s)
    # print(df.append(s, ignore_index=True)) #通过序列添加一行

    #基本操作12：
    # df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
    #                          'foo', 'bar', 'foo', 'bar'],
    #                    'B': ['one', 'one', 'two', 'three',
    #                          'two', 'two', 'one', 'three'],
    #                    'C': np.random.randn(8),
    #                    'D': np.random.randn(8)})
    # print(df)
    #
    # print(df.groupby('A').sum())  #根据A列的索引求和
    # print(df.groupby(['A', 'B']).sum())  #根据A列, 后根据B列的索引求和
    # print(df.groupby(['B', 'A']).sum())  #根据A列, 后根据B列的索引求和

    #基本操作13：
    # tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
    #                      'foo', 'foo', 'qux', 'qux'],
    #                     ['one', 'two', 'one', 'two',
    #                      'one', 'two', 'one', 'two']])) #zip 函数对 对应数据打包成一个个tuple
    # print(tuples)
    # index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    # print(index)
    # df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
    # print(df)
    #
    # stacked = df.stack() #将列索引变成行索引
    # print(stacked)
    # print(stacked.unstack())
    # print(stacked.unstack().unstack())  #转换两次

    #基本操作14：
    # df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
    #                    'B': ['A', 'B', 'C'] * 4,
    #                    'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
    #                    'D': np.random.randn(12),
    #                    'E': np.random.randn(12)})
    # print(df)
    # # 根据A，B索引为行，C的索引为列处理D的值
    # print(pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']))
    #
    # print(df[df.A == 'one'].groupby('C').mean()) #根据A列等于one为索引，根据c列组合的平均值

    # #时间序列1：
    # rng = pd.date_range('20170220', periods=600, freq='S')
    # print(rng)
    # print(pd.Series(np.random.randint(0 ,500,len(rng)), index=rng))

    #时间序列2：
    # rng = pd.date_range('20170220', periods=600, freq='S')
    # print(rng)
    # ts = pd.Series(np.random.randint(0 ,500,len(rng)), index=rng)
    # print(ts.resample('2Min').sum()) #重采样，以2分钟为单位进行加和采样
    #
    # rng1 = pd.period_range('2011Q1', '2017Q1', freq='Q')  #2011第一季度到2017第一季度
    # print(rng1)
    # print(rng1.to_timestamp()) #转成时间戳
    # print(pd.Timestamp('20170220') - pd.Timestamp('20170112'))
    # print(pd.Timestamp('20170220') + pd.Timedelta(days=12))

    # #数据类别
    # df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
    # print(df)
    # df['grade'] = df['raw_grade'].astype('category')  #添加类别数据，以raw_grade的值为类别基础
    # print(df)
    # print(df['grade'].cat.categories) #打印类别
    #
    # df['grade'].cat.categories = ['very good', 'good', 'very bad'] #更改类别
    # print(df)
    #
    # print(df.sort_values(by='grade', ascending=True)) #根据grade的值排序
    # print(df.groupby('grade').size())

    # #数据可视化
    # ts = pd.Series(np.random.randn(1000), index=pd.date_range('20170220', periods=1000))
    # ts = ts.cumsum()
    # print(ts)
    # ts.plot()
    # plt.show()

    # #数据读写
    # df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))
    # #数据保存，相对路径
    # print(df)
    # df.to_csv('data.csv')
    # print(pd.read_csv('data.csv', index_col=0))






















