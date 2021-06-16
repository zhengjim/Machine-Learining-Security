from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from datasets import Datasets
import numpy as np


# 提取特征 向量化 以2-gram
def get_feature(x):
    cv = CountVectorizer(ngram_range=(2, 2), decode_error="ignore", token_pattern=r"\w", min_df=1)
    x = cv.fit_transform(x).toarray()
    return x


def main():
    x, y = Datasets.load_dga_domain()
    x = get_feature(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    print(gnb.score(x_test, y_test))  # 0.9422222222222222

    scores = cross_val_score(gnb, x, y, cv=3, scoring="accuracy")
    print(scores.mean())  # 0.9356666666666666


if __name__ == "__main__":
    main()
