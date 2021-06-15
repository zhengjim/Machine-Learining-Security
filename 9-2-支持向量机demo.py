import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def main():
    # 随机40个点，符号正态分布
    np.random.seed(0)
    x = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
    y = [0] * 20 + [1] * 20

    # 导入svm并训练
    clf = SVC(kernel='linear')
    clf.fit(x, y)

    # 构造超平面
    w = clf.coef_[0]  # 得到w
    a = -w[0] / w[1]  # 找到斜率
    xx = np.linspace(-5, 5)  # -5，5返回均匀间隔的数字
    yy = a * xx - (clf.intercept_[0]) / w[1]  # clf.intercept_[0]#用来获得截距
    b = clf.support_vectors_[0]  # 求出过切线的点
    yy_down = a * xx + (b[1] - a * b[0])  # 下边界
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])  # 上边界

    # matplotlib画图
    # 超平面
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    # 离得最近的向量
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='red')
    # 散点
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)

    plt.axis('tight')
    plt.show()


if __name__ == "__main__":
    main()
