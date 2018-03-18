import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #workfolw
    #step1: 准备数据
    x = [1, 2, 3, 4]
    y = [10, 15, 20, 25]
    #step2: 创建图
    fig = plt.figure()
    #step3： 选择图
    ax = fig.add_subplot(212)
    #step4: 自定义图
    ax.plot(x, y, color='lightblue', linewidth=1)
    # ax.scatter([2, 4, 6], [5, 15, 25], color='darkgreen', marker='^')
    ax.set_xlim(1, 6.5)
    plt.savefig('foo.png')
    plt.show()



    #准备数据
    # x = np.linspace(0, 10, 100)
    # y = np.sin(x)
    # z = np.cos(x)

    # 创建plot
    # fig = plt.figure()
    # fig2 = plt.figure(figsize=plt.figaspect(2))

    # 设置轴线
    # fig.add_axes()
    # ax1 = fig.add_subplot(221) # row-col-num
    # ax3 = fig.add_subplot(212)
    # fig3, axes = plt.subplots(nrows=2, ncols=2)
    # fig4, axes2 = plt.subplots(ncols=3)
    #
    # # plot routines
    # # 1d
    # fig, ax = plt.subplots()

    # lines = ax.plot(x, y)
    # ax.scatter(x, y)
    # axes[0, 0].bar([1, 2, 3], [3, 4, 5])
    # axes[1, 0].barh([0.5, 1, 2.5], [0, 1, 2])
    # axes[1, 1].axhline(0.45)
    # axes[0, 1].axvline(0.65)
    # ax.fill(x, y, color='blue')
    # ax.fill_between(x, y, color='yellow')

    #自定义plot

    #颜色，透明度等属性
    # plt.plot(x, x, x,  x ** 2)
    # ax.plot(x, y, alpha=0.1)
    # ax.plot(x, y, c='k')

    #markers
    # ax.scatter(x, y, marker="o")

    #linstyles
    # plt.plot(x, y, linewidth=4.0)
    # plt.plot(x, y, ls="solid")
    # plt.plot(x, y, ls="-.")
    # plt.plot(x, y, '--', x ** 2, y ** 2, "-.")

    # plt.setp(lines, color='r', linewidth=4.0)

    #文本和注解
    # ax.text(1,
    #         -2.1,
    #         "Example Graph",
    #         style='italic')
    # ax.annotate("Sine",
    #             xy=(8, 0),
    #             xycoords='data',
    #             xytext=(10.5, 0),
    #             textcoords='data',
    #             arrowprops=dict(arrowstyle="->",
    #                             connectionstyle='arc3'
    #                             ),
    #             )
    #标题
    # plt.title("$sigma_i=15$", fontsize=20)

    #范围，图例和布局
    # ax.margins(x=0.0, y=0.1)  #这是x， y的偏移值
    # ax.axis('equal')  #x,y 轴范围
    # ax.set(xlim=[0 , 10.5], ylim=[-1.5, 1.5] )  # 分别设置x，y轴范围

    #legends
    # ax.set(title="An Example Axes", xlabel="x-Axis", ylabel="y-Axis")   # 设置x， y轴名称

    #ticks

    # # save
    # plt.savefig("1.png", transparent=True)  #保存图片，透明可选
    #
    # plt.show()
    #
    #
    #
    # #clear and close
    #
    # plt.cla()
    # plt.clf()
    # plt.close()