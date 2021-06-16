import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def show_dbscan():
    # 生成测试样本
    centers = [[1, 1], [-1, -1], [1, -1]]
    x, y = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
    x = StandardScaler().fit_transform(x)

    # 聚类，半径0.3 最少样本数数10
    db = DBSCAN(eps=0.3, min_samples=10).fit(x)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # 查看分为几簇
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("模型分为:%d簇" % n_clusters_)

    # 评估模型
    # FMI评价法(需要真实值)
    print("FMI评价法: %0.3f" % metrics.fowlkes_mallows_score(y, labels))

    # Calinski-Harabaz Index评估模型(不需要真实值)
    print("Calinski-Harabaz Index评估法: %0.3f" % metrics.calinski_harabasz_score(x, labels))
    # 轮廓系数法 (不需要真实值)
    print("轮廓系数法: %0.3f" % metrics.silhouette_score(x, labels))

    # 画图 噪音用黑色点
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'
        class_member_mask = (labels == k)
        xy = x[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        xy = x[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


if __name__ == '__main__':
    show_dbscan()
