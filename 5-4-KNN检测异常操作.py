import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from datasets import Datasets


# 特征提取，使用词集将操作命令向量化，根据操作统计命令词集来判断
def get_feature(cmd, fdist):
    feature = []
    for block in cmd:
        v = [0] * len(fdist)
        for i in range(0, len(fdist)):
            if fdist[i] in block:
                v[i] += 1
        feature.append(v)
    return feature


def main():
    data, y, fdist = Datasets.load_Schonlau('User3')
    x = get_feature(data, fdist)

    # 训练数据 120  测试数据后30
    # x_train, y_train = x[0:100], y[0:100]
    # x_test, y_test = x[100:150], y[100:150]
    # print(x_test, y_test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # knn训练
    knn = KNeighborsClassifier(n_neighbors=3)
    # knn.fit(x_train, y_train)
    # # 查看模型分数
    # print(knn.score(x_test, y_test))

    # 交叉验证 分10组
    scores = cross_val_score(knn, x, y, cv=10, scoring="accuracy")
    print(scores.mean())  # 0.9733333333333334

    # # 判断k值
    # k_range = range(1, 30)
    # k_scores = []
    # for k in k_range:
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     scores = cross_val_score(knn, x, y, cv=10, scoring="accuracy")
    #     k_scores.append(scores.mean())
    #
    # plt.plot(k_range, k_scores)
    # plt.xlabel("Value of K for KNN")
    # plt.ylabel("Cross Validated Accuracy")
    # plt.show()
    # # 根据图来看 k=3 模型最优


if __name__ == "__main__":
    main()
