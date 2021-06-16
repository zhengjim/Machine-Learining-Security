import numpy as np
from nltk import FreqDist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from datasets import Datasets


# 特征提取，统计该操作序列中的10个与整个数据最频繁使用的前50个命令以及最不频繁使用的前50个命令计算重合程度
def get_feature(cmd, fdist):
    max_cmd = set(fdist[0:50])
    min_cmd = set(fdist[-50:])
    feature = []
    for block in cmd:
        f1 = len(set(block))
        fdist = list(FreqDist(block).keys())
        f2 = fdist[0:10]
        f3 = fdist[-10:]
        f2 = len(set(f2) & set(max_cmd))
        f3 = len(set(f3) & set(min_cmd))
        x = [f1, f2, f3]
        feature.append(x)
    return feature


def main():
    data, y, fdist = Datasets.load_Schonlau('User3')
    # 特征提取
    x = get_feature(data, fdist)
    # 训练数据 120  测试数据后30
    # x_train, y_train = x[0:100], y[0:100]
    # x_test, y_test = x[100:150], y[100:150]
    # print(x_test, y_test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # knn训练
    # knn = KNeighborsClassifier()
    # knn.fit(x_train, y_train)
    # # 查看模型分数
    # print(knn.score(x_test, y_test))
    #
    # # 交叉验证 分10组
    # scores = cross_val_score(knn, x, y, cv=10, scoring="accuracy")
    # print(scores.mean())
    # 判断k值
    k_range = range(1, 30)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x, y, cv=10, scoring="accuracy")
        k_scores.append(scores.mean())

    plt.plot(k_range, k_scores)
    plt.xlabel("Value of K for KNN")
    plt.ylabel("Cross Validated Accuracy")
    plt.show()
    # 根据图来看 k=3 模型最优 约96%


if __name__ == "__main__":
    main()
