from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from datasets import Datasets


def main():
    train_data, vaild_data, test_data = Datasets.load_mnist()
    x_train, y_train = train_data
    x_test, y_test = test_data
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    print(gnb.score(x_test, y_test))  # 0.5544

    scores = cross_val_score(gnb, x_test, y_test, cv=3, scoring="accuracy")
    print(scores.mean())  # 0.5752038611179655


if __name__ == "__main__":
    main()
