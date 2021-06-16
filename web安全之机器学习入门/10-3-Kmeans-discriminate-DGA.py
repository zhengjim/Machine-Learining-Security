from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import calinski_harabasz_score, fowlkes_mallows_score, silhouette_score
from sklearn.manifold import TSNE
from exts import load_alexa, load_dga
import matplotlib.pyplot as plt
import numpy as np


# 提取特征 向量化 以2-gram
def get_feature(x):
    cv = CountVectorizer(ngram_range=(2, 2), decode_error="ignore", token_pattern=r"\w", min_df=1)
    x = cv.fit_transform(x).toarray()
    return x


def main():
    aleax = load_alexa('data/domain/top-1000.csv')
    cry = load_dga('data/domain/dga-cryptolocke-1000.txt')
    goz = load_dga('data/domain/dga-post-tovar-goz-1000.txt')

    x = np.concatenate((aleax, cry, goz))
    x = get_feature(x)

    y = np.concatenate(([0] * len(aleax), [1] * len(cry), [1] * len(goz)))

    # 用SVM模型并训练
    kmeans = KMeans(n_clusters=2, random_state=170)
    kmeans.fit(x)
    labels = kmeans.labels_

    # FMI评价法(需要真实值)
    print(fowlkes_mallows_score(y, labels))

    # 轮廓系数法 (不需要真实值)
    print(silhouette_score(x, labels))

    # Calinski-Harabaz Index评估模型(不需要真实值)
    print(calinski_harabasz_score(x, labels))

    # 数据降维与可视化(慢、占内存)
    tsne = TSNE(learning_rate=100)
    x = tsne.fit_transform(x)
    for i, label in enumerate(x):
        x1, x2 = x[i]
        if labels[i] == 1:
            plt.scatter(x1, x2, marker='o')
        else:
            plt.scatter(x1, x2, marker='x')
    plt.show()


if __name__ == "__main__":
    main()
