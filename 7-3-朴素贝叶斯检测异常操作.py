from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pydotplus
from datasets import Datasets


# 特征提取，使用词集将操作命令向量化
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
    # 特征提取
    x = get_feature(data, fdist)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 朴素贝叶斯
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    print(gnb.score(x_test, y_test))  # 1.0

    # 交叉验证
    scores = cross_val_score(gnb, x, y, cv=10, scoring="accuracy")
    print(scores.mean())  # 0.9933333333333334


if __name__ == "__main__":
    main()
