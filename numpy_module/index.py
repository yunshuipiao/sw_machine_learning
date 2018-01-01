import numpy as np
import math

if __name__ == '__main__':

    # ----------------------
    # a = np.zeros((3, 4))
    # b = np.ones((3, 4), dtype=np.int16)
    # c = np.arange(10, 25, 5)
    # d = np.linspace(0, 2, 9)
    # #0～2之间均分9个数字：[ 0.    0.25  0.5   0.75  1.    1.25  1.5   1.75  2.  ]
    # e = np.full((2, 2), 7)
    # f = np.eye(3)  # 单位矩阵
    # g = np.random.random((3, 3))  # 0~1之间的随机矩阵
    # h = np.empty((3, 2))

    # ----------------------
    #saving or loading on Disk
    # np.save('my_array', b)
    # np.savez('array.npz', a, b, c)
    #
    # data = np.load('array.npz')
    # print(data['arr_2'])

    # ----------------------
    #text or csv file
    # np.savetxt('myarray.txt', c, delimiter=" ")
    # data = np.loadtxt("myarray.txt")
    # print(data)

    # ----------------------
    # np的几个属性  inspect
    # j = a
    # print(j)
    # print( j.shape)       #array的维度描述
    # print(len(j) )        #array的x（行）的长度
    # print(j.ndim)         #array维度
    # print(j.size)         #元素个数
    #
    # print(j.dtype)        #元素类型
    # print(j.dtype.name)   #元素类型名字
    # print(b.astype(int))  #元素类型转换

    #for help
    # print(np.info(np.ndarray.dtype))

    # ----------------------
    #简单运算
    a = np.array([(1, 2, 3)])
    b = np.array([(1.5, 2, 3), (4, 5, 6)], dtype=float)
    c = np.array([[(1.5, 2, 3), (4, 5, 6)], [(3, 2, 1), (4, 5, 6)]])
    # print(a - b)
    # print(np.subtract(a, b))  #两者一样
    #
    # print(a + b)
    # print(np.add(a , b))

    # print(b / a)
    # print(np.divide(a, b))

    # print(a * b)
    # print(np.multiply(a, b))

    # print(np.exp([1, 2, 1]))  #计算每个元素e为底， 元素为幂的值。
    # print(np.sqrt(a))
    # print(np.sin(a))
    # print(np.cos(a))
    # print(np.log(a))  # 幂为底的对数值

    # print(a.dot(b))   #矩阵点乘，需满足条件

    # ----------------------
    #compare
    # print(a == b)  #元素之间的比较
    # print(a < 2)

    # ----------------------
    # aggregate 集合运算
    # print(b.sum())
    # print(b.min())
    # print(b.max(axis=0))  #axis指定标准，0表示row
    # print(b.cumsum(axis=1)) #累计和，当前元素加上前面的和
    # print(a.mean())
    # print(np.std(a))  #计算标准差

    # ----------------------
    # array copy
    # h = a.view()  #创建视图，修改则会影响到源数据
    # print(np.copy(a))
    # print(a.copy())   # copy则不会

    # ----------------------
    #sort
    # print(b.sort(axis=0))

    # ----------------------
    # 子数据，切片和索引
    # print(b[1][1])  #b[1, 1]
    # print(b[0: 1, 1: 3])  #(行，列)
    # print(b[:1])  #选择每列的所有item  b[0:1, :]
    # print(c[1, ...])  # #c[1, :, :]
    #反转矩阵，-1表示反转
    # d = b[::1]  #行
    # d = b[::1, ::-1]  #列
    # d = b[::-1, ::-1]
    # print(d)
    # print(b[b<5])

    # ----------------------
    # 矩阵处理
    # i = np.transpose(b)  #矩阵行列变换  b.T
    # print(i)
    # print(b.T)

    # i = b.ravel()  # 变矩阵为一维矩阵
    # ii = b.reshape(3, -2)  #reshape矩阵， (行，列), 负数自动计算

    # g = b.reshape(3, -2)
    # h = b.copy()
    # h.resize(2, 6)   #reshape不改变矩阵，而resize改变矩阵，不足用0补全
    # print(np.append(h, g))
    # print(np.insert(a, 1, 5))
    # print(np.delete(a, [1]))

    # print(np.hsplit(a, 3))  #分割矩阵  水平分割
    # print(np.vsplit(b, 2))




