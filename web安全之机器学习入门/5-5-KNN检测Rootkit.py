import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from datasets import Datasets
import matplotlib.pyplot as plt
import pandas as pd


def get_rootkit_and_normal(kdd99_data):
    x = []
    y = []
    for data in kdd99_data:
        if (data[41] in ["normal.", "rootkit."]) and (data[2] == "telnet"):
            # 提取结果
            y.append(1 if data[41] == "rootkit." else 0)
            # 提取特征
            x.append(list(map(lambda i: float(i), data[9:21])))
    return x, y


def main():
    kdd99_data = Datasets.load_kdd99()
    x, y = get_rootkit_and_normal(kdd99_data)

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    knn = KNeighborsClassifier(n_neighbors=3)

    # knn.fit(x_train, y_train)
    # print(knn.score(x_test, y_test))
    scores = cross_val_score(knn, x, y, cv=2, scoring='accuracy')
    print(scores.mean())  # 0.9777777777777777


if __name__ == "__main__":
    main()
