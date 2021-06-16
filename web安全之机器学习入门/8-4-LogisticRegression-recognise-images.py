from datasets import Datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def main():
    # 加载MNIST数据
    train_data, valid_data, test_data = Datasets.load_mnist()
    x_train, y_train = train_data
    x_test, y_test = test_data

    # 逻辑回归训练并预测
    lr = LogisticRegression(solver='lbfgs', max_iter=2000)
    lr.fit(x_train, y_train)
    print(lr.score(x_test, y_test))

    scores = cross_val_score(lr, x_test, y_test, cv=10, scoring="accuracy")
    print(scores.mean())


if __name__ == "__main__":
    main()
