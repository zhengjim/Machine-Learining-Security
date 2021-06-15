import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm


def main():
    # 初始概率矩阵
    startprob = np.array([0.6, 0.3, 0.1, 0.0])
    # 转移概率矩阵
    transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                         [0.3, 0.5, 0.2, 0.0],
                         [0.0, 0.3, 0.5, 0.2],
                         [0.2, 0.0, 0.2, 0.6]])
    # 每个成分的均值
    means = np.array([[0.0, 0.0],
                      [0.0, 11.0],
                      [9.0, 10.0],
                      [11.0, -1.0]])
    # 每个成分的协方差
    covars = .5 * np.tile(np.identity(2), (4, 1, 1))
    # 定义算法参数
    model = hmm.GaussianHMM(n_components=4, covariance_type="full")
    # 我们没有从数据中拟合它，而是直接设置估计值
    # 参数、分量的均值和协方差
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars
    # 生成示例
    X, Z = model.sample(500)
    # 画图
    plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
             mfc="orange", alpha=0.7)

    for i, m in enumerate(means):
        plt.text(m[0], m[1], 'Component %i' % (i + 1),
                 size=17, horizontalalignment='center',
                 bbox=dict(alpha=.7, facecolor='w'))
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
