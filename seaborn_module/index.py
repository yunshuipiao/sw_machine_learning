import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
    plt.show()

if __name__ == '__main__':
    pass

    # sns.set()
    # sinplot()

