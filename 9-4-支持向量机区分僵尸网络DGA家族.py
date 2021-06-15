from sklearn.svm import SVC
from datasets import Datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from exts import load_alexa, load_dga
import matplotlib.pyplot as plt
import numpy as np
import re


# 特征选取： 元音字母的比例、去重后的字母数字个数与域名长度的比例、平均jarccard系数(交集与并集的个数)、HMM系数(隐马尔可夫模型)

# 元音字母的比例 x:域名长度 y:元音字母的比例
def aeiou_count(domain_list):
    x = []
    y = []
    for domain in domain_list:
        x.append(len(domain))
        count = len(re.findall(r'[aeiou]', domain.lower()))
        count = (0.0 + count) / len(domain)
        y.append(count)
    return x, y


# 去重后的字母数字个数与域名长度的比例
def get_uniq_char_num(domain_list):
    x = []
    y = []
    for domain in domain_list:
        x.append(len(domain))
        count = len(set(domain))
        count = (0.0 + count) / len(domain)
        y.append(count)
    return x, y


# 计算两个域名之间的jarccard系数
def count2string_jarccard_index(a, b):
    x = set(' ' + a[0])
    y = set(' ' + b[0])
    for i in range(0, len(a) - 1):
        x.add(a[i] + a[i + 1])
    x.add(a[len(a) - 1] + ' ')
    for i in range(0, len(b) - 1):
        y.add(b[i] + b[i + 1])
    y.add(b[len(b) - 1] + ' ')

    return (0.0 + len(x - y)) / len(x | y)


# 计算两个域名集合的平均jarccard系数
def get_jarccard_index(a_list, b_list):
    x = []
    y = []
    for a in a_list:
        j = 0.0
        for b in b_list:
            j += count2string_jarccard_index(a, b)
        x.append(len(a))
        y.append(j / len(b_list))
    return x, y


# 平均jarccard系数
def jarccard_mean(domain_list):
    x, y = get_jarccard_index(domain_list, aleax)
    return x, y


# 根据特征函数画图
def dradrawing(my_feature):
    x1, y1 = my_feature(aleax)
    x2, y2 = my_feature(cry)
    x3, y3 = my_feature(goz)

    # 画图
    fig, ax = plt.subplots()
    ax.set_xlabel('Domain Length')
    ax.set_ylabel('Score')
    ax.scatter(x3, y3, color='b', label="dga_post-tovar-goz", marker='o')
    ax.scatter(x2, y2, color='g', label="dga_cryptolock", marker='v')
    ax.scatter(x1, y1, color='r', label="alexa", marker='*')
    ax.legend(loc='best')
    plt.show()


# 特征提取
def get_feature(domain_list):
    x = []
    _, x1 = aeiou_count(domain_list)
    _, x2 = get_uniq_char_num(domain_list)
    _, x3 = jarccard_mean(domain_list)

    for i in range(0, len(x1)):
        x.append([x1[i], x2[i], x3[i]])
    return x


def main():
    # 画图
    # dradrawing(aeiou_count)
    # dradrawing(get_uniq_char_num)
    # dradrawing(jarccard_mean)

    # 特征提取
    x = np.concatenate((get_feature(aleax), get_feature(cry), get_feature(goz)))
    y = np.concatenate(([0] * len(aleax), [1] * len(cry), [2] * len(goz)))

    # 标准化
    std = StandardScaler()
    x = std.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

    # 用SVM模型并训练
    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)

    print(clf.score(x_test, y_test))


if __name__ == "__main__":
    aleax = load_alexa('data/domain/top-1000.csv')
    cry = load_dga('data/domain/dga-cryptolocke-1000.txt')
    goz = load_dga('data/domain/dga-post-tovar-goz-1000.txt')
    main()
