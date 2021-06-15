import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def show_kmeans():
    # 生成测试样本
    n_samples = 1500
    random_state = 170
    x, y = make_blobs(n_samples=n_samples, random_state=random_state)

    # 聚类，指定聚类个数为3
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(x)

    # 画图
    plt.subplot(221)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.title("K-means")

    plt.show()


if __name__ == '__main__':
    show_kmeans()
