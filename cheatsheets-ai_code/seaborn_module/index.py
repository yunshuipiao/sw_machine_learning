import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd


def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
    plt.show()

if __name__ == '__main__':
    # fig, ax = plt.subplots()
    #
    # tips = sns.load_dataset("tips")
    # print(tips)
    # sns.violinplot(x="total_bill", data=tips)
    # plt.show()
    data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
    print(data)