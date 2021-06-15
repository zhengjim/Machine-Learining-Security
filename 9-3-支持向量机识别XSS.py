from sklearn.svm import SVC
from datasets import Datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import re


# 特征选取： url长度、url包含第三方域名个数、敏感字符个数、敏感关键字个数

# url长度
def url_len(url):
    return len(url)


# url是否包含第三方域名
def url_has_domain(url):
    return 1 if re.search('(http://)|(https://)', url, re.IGNORECASE) else 0


# 敏感字符个数
def evil_str_count(url):
    return len(re.findall("[<>,\'\"/]", url, re.IGNORECASE))


# 敏感关键字个数
def evil_keywords_count(url):
    blacklist = "(alert)|(script=)(%3c)|(%3e)|(%20)|(onerror)|(onload)|(eval)|(src=)|(prompt)"
    return len(re.findall(blacklist, url, re.IGNORECASE))


# 特征提取
def get_feature(url):
    return [url_len(url), url_has_domain(url), evil_str_count(url), evil_keywords_count(url)]


def main():
    data, y = Datasets.load_xss()
    x = []
    for url in data:
        x.append(get_feature(url))

    # 标准化
    std = StandardScaler()
    x = std.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

    # 用SVM模型并训练
    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)

    print(clf.score(x_test, y_test))

    # 交叉验证 十组比较慢
    scores = cross_val_score(clf, x, y, cv=10, scoring='accuracy')
    print(scores.mean())


if __name__ == "__main__":
    main()
