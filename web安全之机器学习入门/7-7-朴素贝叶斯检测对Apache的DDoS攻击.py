from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from datasets import Datasets


def get_apache_ddos_and_normal(kdd99_data):
    x = []
    y = []
    for data in kdd99_data:
        if (data[41] in ["normal.", "apache2."]) and (data[2] == "http"):
            # 提取结果
            y.append(1 if data[41] == "apache2." else 0)
            # 提取特征
            x.append(list(map(lambda i: float(i), [data[0]] + data[4:8] + data[22:30])))
    return x, y


def main():
    kdd99_data = Datasets.load_kdd99()
    x, y = get_apache_ddos_and_normal(kdd99_data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    print(gnb.score(x_test, y_test))  # 0.9972529759427287

    scores = cross_val_score(gnb, x, y, cv=3, scoring="accuracy")
    print(scores.mean())  # 0.9965785070302938


if __name__ == "__main__":
    main()
