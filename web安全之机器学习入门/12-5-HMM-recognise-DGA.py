import numpy as np
from hmmlearn import hmm
from exts import load_alexa, load_dga
import matplotlib.pyplot as plt
import joblib

# 处理参数值的最小长度
MIN_LEN = 6
# 隐状态个数
N = 10
# 最大似然概率阈值
T = -200


# 数据预处理
def get_feature(domain):
    ver = []
    for i in range(0, len(domain)):
        ver.append([ord(domain[i])])
    return ver


# 训练
def train(domain_list):
    x = [[0]]
    x_lens = [1]
    for domain in domain_list:
        ver = get_feature(domain)
        np_ver = np.array(ver)
        x = np.concatenate([x, np_ver])
        x_lens.append(len(np_ver))

    ghmm = hmm.GaussianHMM(n_components=N, covariance_type="full", n_iter=100)
    ghmm.fit(x, x_lens)
    joblib.dump(ghmm, "export/model/hmm-dga-train.pkl")

    return ghmm


def test(filename, flag):
    ghmm = joblib.load("export/model/hmm-dga-train.pkl")

    x = []
    y = []
    domain_list = load_alexa(filename) if flag == 1 else load_dga(filename)
    for domain in domain_list:
        domain_ver = get_feature(domain)
        np_ver = np.array(domain_ver)
        pro = ghmm.score(np_ver)
        x.append(len(domain))
        y.append(pro)

    return x, y


def main():
    # 加载数据
    aleax = load_alexa('data/domain/top-1000.csv')
    # 训练
    train(aleax)

    x1, y1 = test('data/domain/dga-post-tovar-goz-1000.txt', 0)
    x2, y2 = test('data/domain/dga-cryptolocke-1000.txt', 0)
    x3, y3 = test('data/domain/top-1000.csv', 1)

    # 画图
    fig, ax = plt.subplots()
    ax.set_xlabel('Domain Length')
    ax.set_ylabel('HMM Score')
    ax.scatter(x3, y3, color='b', label="dga_post-tovar-goz")
    ax.scatter(x2, y2, color='g', label="dga_cryptolock")
    ax.scatter(x1, y1, color='r', label="alexa")
    ax.legend(loc='right')
    plt.show()


if __name__ == '__main__':
    main()
