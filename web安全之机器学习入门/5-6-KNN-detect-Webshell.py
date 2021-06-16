import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from datasets import Datasets
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # 加载ADFA-LD 数据
    x1, y1 = Datasets.load_adfa_normal()
    x2, y2 = Datasets.load_adfa_attack(r"Web_Shell_\d+/UAD-W*")
    x = x1 + x2
    y = y1 + y2

    # 词袋特征
    cv = CountVectorizer()
    x = cv.fit_transform(x).toarray()

    knn = KNeighborsClassifier(n_neighbors=3)

    scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
    print(scores.mean())  # 0.9663706140350878


if __name__ == "__main__":
    main()
