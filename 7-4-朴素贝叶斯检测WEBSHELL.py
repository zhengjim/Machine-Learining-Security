from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from datasets import Datasets
import numpy as np


# 数据预处理 向量化 以2-gram
def to_voc(webshell, wordpress):
    webshell_voc = CountVectorizer(ngram_range=(2, 2), decode_error="ignore", token_pattern=r'\b\w+\b', min_df=1)
    x1 = webshell_voc.fit_transform(webshell).toarray()
    y1 = [1] * len(x1)

    wordpress_voc = CountVectorizer(ngram_range=(2, 2), decode_error="ignore", token_pattern=r'\b\w+\b', min_df=1,
                                    vocabulary=webshell_voc.vocabulary_)
    x2 = wordpress_voc.fit_transform(wordpress).toarray()
    y2 = [0] * len(x2)

    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))

    return x, y


def main():
    webshell, wordpress = Datasets.load_php_webshell()
    x, y = to_voc(webshell, wordpress)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    print(gnb.score(x_test, y_test))  # 0.7659574468085106

    scores = cross_val_score(gnb, x, y, cv=3, scoring="accuracy")
    print(scores.mean())  # 0.7872046254399195


if __name__ == "__main__":
    main()
