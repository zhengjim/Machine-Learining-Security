from model.perceptron import Perceptron
from model.linearunit import LinearUnit
from exts import *
import matplotlib.pyplot as plt


def perceptron_test():
    """感知器"""
    # 实现and函数，基于and真值表构建训练数据
    x = [[0, 0], [1, 0], [0, 1], [1, 1]]
    y = [0, 0, 0, 1]

    # 创建感知器，输入参数为2（因为and是二元函数），激活函数为relu
    p = Perceptron(2, relu)
    # 训练5次，学习速率为0.1
    p.fix(x, y, 5, 0.1)
    print(p)
    # 验证模型
    print("0 and 0 = %d" % p.predict([0, 0]))
    print("0 and 1 = %d" % p.predict([0, 1]))
    print("1 and 0 = %d" % p.predict([1, 0]))
    print("1 and 1 = %d" % p.predict([1, 1]))


def linear_test():
    """线性单元"""
    # 生成5个人的收入数据
    x = [[5], [3], [8], [1.4], [10.1], [2]]
    y = [5500, 2300, 7600, 1800, 11400, 2000]

    # 创建感知器，线性单元
    lu = LinearUnit(1)
    lu.fix(x, y, 10, 0.01)
    print(lu)
    print(lu.predict([3.4]))

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(list(map(lambda x1: x1[0], x)), y)
    weights = lu.w
    bias = lu.b
    x = range(0, 12, 1)
    y = list(map(lambda x: weights[0] * x + bias, x))
    ax.plot(x, y)
    plt.show()


def main():
    # 感知器
    perceptron_test()
    # 线性单元
    linear_test()


if __name__ == '__main__':
    main()
