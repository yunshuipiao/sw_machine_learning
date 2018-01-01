
import pandas as pd

if __name__ == '__main__':

    #--------------
    # syntax
    # df = pd.DataFrame({
    #     "a": [4, 5, 6],
    #     "b": [7, 8, 9],
    #     "c": [10, 11, 12]
    # }, index=[1, 2, 3])  #index 指定每一行的名称

    df = pd.DataFrame([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], columns=["a", "b", "c"])  #两种方式，指定行列名称。
    print(df)

    # reshape data
    print(pd.melt(df))
    print(pd.concat([df, df], axis=1))




