import numpy as np
from hmmlearn import hmm
from urllib.parse import unquote, parse_qsl
from html import unescape
import re
import joblib

# 以白找黑(基本无漏报、但误报多)
# 优点：理论上可以发现全部基于参数的异常访问。
# 缺点：扫描器访问、代码异常、用户的错误操作、业务代码的升级等，都会产生大量误报

# 处理参数值的最小长度
MIN_LEN = 6
# 隐状态个数
N = 10
# 最大似然概率阈值
T = -50
# 字母
# 数字 1
# <>,:"'
# 其他字符2
SEN = ['<', '>', ',', ':', '\'', '/', ';', '"', '{', '}', '(', ')']


# #排除中文干扰 只处理127以内的字符
def ischeck(str1):
    if re.match(r'^(http)', str1):
        return False
    for i, c in enumerate(str1):
        if ord(c) > 127 or ord(c) < 31:
            return False
        if c in SEN:
            return True
    return True


# 特征离散规范化 按照不同域名的不同url的不同参数分别学习
def etl(str1):
    vers = []
    for i, c in enumerate(str1):
        c = c.lower()
        if ord(c) >= ord('a') and ord(c) <= ord('z'):
            vers.append([ord(c)])
        elif ord(c) >= ord('0') and ord(c) <= ord('9'):
            vers.append([1])
        elif c in SEN:
            vers.append([ord(c)])
        else:
            vers.append([2])
    return np.array(vers)


# 特征提取
def generate_feature(filename):
    x = [[0]]
    x_lens = [1]
    sample = [['']]
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            # url解码
            line = unquote(line)
            line = unescape(line)
            if len(line) >= MIN_LEN:
                params = parse_qsl(line, True)
                for k, v in params:
                    if ischeck(v) and len(v) >= N:
                        vers = etl(v)  # 数据处理与特征提取
                        x = np.concatenate([x, vers])  # 每一个参数value作为一个特征向量
                        x_lens.append(len(vers))  # 长度
                        sample.append(v)
    return x, x_lens, sample


def train(filename):
    x, x_lens, _ = generate_feature(filename)
    ghmm = hmm.GaussianHMM(n_components=N, covariance_type="full", n_iter=100)
    ghmm.fit(x, x_lens)
    # 训练时间比较久 保存一份
    joblib.dump(ghmm, "export/model/hmm-xss-train.pkl")
    return ghmm


def test(filename):
    # 从pkl读取
    ghmm = joblib.load("export/model/hmm-xss-train.pkl")
    x, x_lens, sample = generate_feature(filename)
    x = x[1:, :]
    x_lens = x_lens[1:]
    lastindex = 0
    for i in range(len(x_lens)):
        line = np.array(x[lastindex:lastindex + x_lens[i], :])
        lastindex += x_lens[i]
        pro = ghmm.score(line)
        if pro < T:
            print("SCORE:(%d) Sample: %s" % (pro, sample[i]))


def main():
    # 以白找黑， 训练白样本。让模型记住正常url参数的转化概率
    # ghmm = train("data/web-attack/normal-10000.txt")

    # 输入test样本检测，设定一个阈值(score的分值越小，说明出现的概率越小，也即说明样本的偏离正常程度)
    test('data/XSS/test-sample.txt')


if __name__ == '__main__':
    main()
