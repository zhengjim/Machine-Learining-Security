import os
import csv
import random
import numpy as np


def load_one_flle(filename):
    """单文件加载

    :param filename: 文件名
    :return:
    """
    with open(filename) as f:
        line = f.readline()
        line = line.strip('\n')
    return line


def load_files(path):
    """加载目录下全部文件,除去加载失败的

    :param path:
    :return:
    """
    files_list = []
    for r, d, files in os.walk(path):
        for file in files:
            if file.endswith('.php'):
                file_path = path + file
                t = load_file(file_path)
                if t:
                    files_list.append(t)
    return files_list


def load_file(file_path):
    """文件加载并拼接成一个字符串

    :param file_path:
    :return:
    """
    t = ""
    try:
        with open(file_path) as f:
            for line in f:
                t += line
        return t
    except Exception as e:
        return None


def dir_list(path, all_file):
    """递归加载目录全部文件

    :param path:  路径
    :param all_file: 文件列表
    :return: 文件列表
    """
    file_list = os.listdir(path)
    for filename in file_list:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dir_list(filepath, all_file)
        else:
            all_file.append(filepath)
    return all_file


def load_alexa(filename):
    """加载alexa文件

    :param filename:
    :return:
    """
    domain_list = []
    csv_reader = csv.reader(open(filename))
    for row in csv_reader:
        domain_list.append(row[1])
    return domain_list


def load_dga(filename):
    """加载dga文件

    :param filename:
    :return:
    """
    domain_list = []
    with open(filename) as f:
        for line in f:
            domain = line.split(",")[0]
            domain_list.append(domain)
    return domain_list


def load_filename(filename, flag):
    """加载文件内容并给目标值

    :param filename:
    :param flag:
    :return:
    """
    x = []
    y = []
    with open(filename) as f:
        for line in f:
            line = line.strip("\n")
            x.append(line)
            y.append(flag)
    return x, y


def get_one_hot(x, size=10):
    """one-hot编码

    :param x:
    :param size: 大小
    :return:
    """
    v = []
    for x1 in x:
        x2 = [0] * size
        x2[(x1 - 1)] = 1
        v.append(x2)
    return v


def load_files_lable(rootdir, label):
    """加载目录下文件并打标签

    :param rootdir: 路径
    :param label: 标签
    :return:
    """
    dir_list = os.listdir(rootdir)
    x = []
    y = []
    for i in range(0, len(dir_list)):
        path = os.path.join(rootdir, dir_list[i])
        if os.path.isfile(path):
            if load_file(path) is not None:
                x.append(load_file(path))
                y.append(label)
    return x, y


def string_to_semi_redundant_sequences(string, seq_maxlen=25, redun_step=3, char_idx=None):
    """对字符串进行矢量化，并返回解析的序列和目标以及相关字典。

    :param string: 输入文本
    :param seq_maxlen: 序列的最大长度。默认值：25。
    :param redun_step: 冗余步骤。默认值：3。
    :param char_idx: 把字符转换成位置的字典。如果没有，将自动生成

    :return: x，y，字典
    """

    if char_idx is None:
        chars = set(string)
        # 如果对同一个字符集再次运行，sorted将尝试保持字典的一致性
        char_idx = {c: i for i, c in enumerate(sorted(chars))}

    len_chars = len(char_idx)

    sequences = []
    next_chars = []
    for i in range(0, len(string) - seq_maxlen, redun_step):
        sequences.append(string[i: i + seq_maxlen])
        next_chars.append(string[i + seq_maxlen])

    X = np.zeros((len(sequences), seq_maxlen, len_chars), dtype=np.bool)
    Y = np.zeros((len(sequences), len_chars), dtype=np.bool)
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_idx[char]] = 1
        Y[i, char_idx[next_chars[i]]] = 1

    print("Text total length: {:,}".format(len(string)))
    print("Distinct chars   : {:,}".format(len_chars))
    print("Total sequences  : {:,}".format(len(sequences)))

    return X, Y, char_idx


def random_sequence_from_string(string, seq_maxlen):
    """从字符串随机返回一个向量

    :param string: 字符串
    :param seq_maxlen:序列的最大长度
    :return:
    """

    rand_index = random.randint(0, len(string) - seq_maxlen - 1)

    return string[rand_index: rand_index + seq_maxlen]


def sample(preds, temperature=1.0):
    """从预测集中采样索引的辅助函数

    :param preds: 预测集
    :param temperature: 新颖程度
    :return:
    """
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)
