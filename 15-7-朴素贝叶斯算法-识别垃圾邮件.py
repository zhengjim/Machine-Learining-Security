from datasets import Datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB


# 朴素贝叶斯
def main():
    x, y = Datasets.load_spambase()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

    # 训练
    gnb = GaussianNB()
    # gnb.fit(x_train, y_train)

    # print(gnb.score(x_test, y_test))  # 0.8204199855177408

    # 交叉验证
    sorces = cross_val_score(gnb, x, y, cv=10, scoring="accuracy")
    print(sorces.mean())  # 0.8217730830896915


if __name__ == "__main__":
    main()
