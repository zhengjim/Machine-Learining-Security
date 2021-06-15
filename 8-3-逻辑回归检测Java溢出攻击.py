from datasets import Datasets
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score


def main():
    # 加载ADFA-LD 数据
    x1, y1 = Datasets.load_adfa_normal()
    x2, y2 = Datasets.load_adfa_attack(r"Java_Meterpreter_\d+/UAD-Java-Meterpreter*")
    x = x1 + x2
    y = y1 + y2

    # 词袋特征
    cv = CountVectorizer(min_df=1)
    x = cv.fit_transform(x).toarray()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

    # 逻辑回归训练并预测
    lr = LogisticRegression(solver='lbfgs', max_iter=2000)
    lr.fit(x_train, y_train)
    print(lr.score(x_test, y_test))  # 0.9340277777777778

    scores = cross_val_score(lr, x, y, cv=10, scoring="accuracy")
    print(scores.mean())  # 0.9498574561403508


if __name__ == "__main__":
    main()
