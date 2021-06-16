import numpy as np
from hmmlearn import hmm
from urllib.parse import unquote, parse_qsl
import re
import joblib
import nltk

# 以黑找黑

# 处理参数值的最小长度
MIN_LEN = 6
# 隐状态个数
N = 10
# 最大似然概率阈值
T = -200

# 数据提取与特征提取，这一步不采用单字符的char特征提取，而是根据领域经验对特定的phrase字符组为基础单位，进行特征化，这是一种token切分方案
# </script><script>alert(String.fromCharCode(88,83,83))</script>
# <IMG SRC=x onchange="alert(String.fromCharCode(88,83,83))">
# <;IFRAME SRC=http://ha.ckers.org/scriptlet.html <;
# ';alert(String.fromCharCode(88,83,83))//\';alert(String.fromCharCode(88,83,83))//";alert(String.fromCharCode(88,83,83))
# //\";alert(String.fromCharCode(88,83,83))//--></SCRIPT>">'><SCRIPT>alert(String.fromCharCode(88,83,83))</SCRIPT>


tokens_pattern = r'''(?x)
 "[^"]+" #"xxxx"
|http://\S+ #http://xxxx
|</\w+> #</xxx>
|<\w+> #<xxx>
|<\w+ #<xxxx
|\w+= # xxxx=
|> # >
|\w+\([^<]+\) #函数 比如alert(String.fromCharCode(88,83,83))
|\w+ # xxxx
'''


# #排除中文干扰 只处理127以内的字符
def ischeck(str1):
    for i, c in enumerate(str1):
        if ord(c) > 127 or ord(c) < 31:
            return False
    return True


# 数据预处理
def preprocessing(str1):
    result = []
    line = str1.strip('\n')
    line = unquote(line)  # url解码
    if len(line) >= MIN_LEN:  # 忽略短url value
        # 只处理参数
        params = parse_qsl(line, True)
        for k, line in params:
            if ischeck(line) and len(line) >= N:
                line, _ = re.subn(r'\d+', "8", line)  # 数字常量替换成8
                line, _ = re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?:=]+', "http://u", line)  # url换成http://u
                line, _ = re.subn(r'\/\*.?\*\/', "", line)  # 去除注释
                tokens = nltk.regexp_tokenize(line, tokens_pattern)  # token切分
                result += tokens
        if result:
            return result
    return False


# 加载词集
def load_wordbag(filename, max=100):
    tokens_list = []
    index_wordbag = 1  # 词袋索引
    wordbag = {}  # 词袋

    with open(filename) as f:
        for line in f:
            tokens = preprocessing(line)
            if tokens:
                tokens_list += tokens
    fredist = nltk.FreqDist(tokens_list)  # 单文件词频
    keys = list(fredist.keys())
    keys = keys[:max]  # 降维，只提取前N个高频使用的单词，其余规范化到0
    for localkey in keys:  # 获取统计后的不重复词集
        if localkey in wordbag.keys():  # 判断该词是否已在词袋中
            continue
        else:
            wordbag[localkey] = index_wordbag
            index_wordbag += 1
    return wordbag


# 训练
def train(filename, wordbag):
    x = [[-1]]
    x_lens = [1]

    with open(filename) as f:
        for line in f:
            words = preprocessing(line)
            if words:
                vers = []
                for word in words:
                    # 根据词汇编码表进行index编码，对不在词汇表中的token词不予编码
                    if word in wordbag.keys():
                        vers.append([wordbag[word]])
                    else:
                        vers.append([-1])

            np_vers = np.array(vers)
            x = np.concatenate([x, np_vers])
            x_lens.append(len(np_vers))

    ghmm = hmm.GaussianHMM(n_components=N, covariance_type="full", n_iter=100)
    ghmm.fit(x, x_lens)
    joblib.dump(ghmm, "export/model/hmm-xss-train_2.pkl")

    return ghmm


# 测试
def test(filename, wordbag):
    # 从pkl读取
    ghmm = joblib.load("export/model/hmm-xss-train_2.pkl")
    with open(filename) as f:
        for line in f:
            words = preprocessing(line)
            if words:
                vers = []
                for word in words:
                    # test和train保持相同的编码方式
                    if word in wordbag.keys():
                        vers.append([wordbag[word]])
                    else:
                        vers.append([-1])
                np_vers = np.array(vers)
                pro = ghmm.score(np_vers)
                if pro >= T:
                    print("SCORE:(%d) XSS_URL: %s " % (pro, line))


def main():
    xss = "data/XSS/xss-200000.txt"
    # 得到词频编码表
    wordbag = load_wordbag(xss, 2000)

    # 以黑找黑, 训练HMM模型, 保存
    train(xss, wordbag)

    # 输入test样本检测
    test('data/XSS/test-sample.txt', wordbag)


if __name__ == '__main__':
    main()
