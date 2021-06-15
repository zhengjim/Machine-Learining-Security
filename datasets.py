from exts import *
from nltk import FreqDist
import numpy as np
import pandas as pd
import os
import re
import gzip
import pickle


class Datasets:
    """加载数据集"""

    @staticmethod
    def load_Schonlau(user):
        """加载Masqera Schonlau数据集，导入用户的操作
        加载数据集，100命令为1个序列、共150个。共15000个命令。前5000都正常命令、后10000有恶意命令。

        :param user:  导入哪个用户的操作
        :return:
        """
        with open("data/MasqueradeDat/" + user) as f:
            lines = f.readlines()
        i = 0
        x = []
        all_cmd = []
        data = []
        for line in lines:
            line = line.strip('\n')
            x.append(line)
            all_cmd.append(line)
            i += 1
            if i == 100:
                data.append(x)
                x = []
                i = 0
        fdist = list(FreqDist(all_cmd).keys())

        # 加载labels
        index = int(user[-1]) - 1
        y = []
        with open("data/MasqueradeDat/label.txt") as f:
            for line in f:
                line = line.strip('\n')
                y.append(int(line.split()[index]))
        y = [0] * 50 + y

        return data, y, fdist

    @staticmethod
    def load_kdd99():
        """加载KDD99数据集"""
        with open('data/kddcup99/corrected') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip('\n')
            line = line.split(",")
            data.append(line)

        return data

    @staticmethod
    def load_adfa_normal():
        """加载ADFA-LD 正常数据集"""
        x = []
        y = []
        list = os.listdir('data/ADFA-LD/Training_Data_Master/')
        for i in range(0, len(list)):
            path = os.path.join('data/ADFA-LD/Training_Data_Master/', list[i])
            if os.path.isfile(path):
                x.append(load_one_flle(path))
                y.append(0)

        return x, y

    @staticmethod
    def load_adfa_attack(reg):
        """加载ADFA-LD 攻击数据集

        :param reg: 攻击类型文件 正则
        :return:
        """
        x = []
        y = []
        all_file = dir_list("data/ADFA-LD/Attack_Data_Master/", [])
        for file in all_file:
            if re.match("data/ADFA-LD/Attack_Data_Master/" + reg, file):
                x.append(load_one_flle(file))
                y.append(1)

        return x, y

    @staticmethod
    def load_php_webshell():
        """加载phpwebshell数据集，正常数据用wordpress"""
        webshell = load_files("data/PHP-WEBSHELL/xiaoma/")
        wordpress = load_files("data/wordpress/")

        return webshell, wordpress

    @staticmethod
    def load_dga_domain():
        """加载dga数据集，正常数据区alexa top1000"""
        aleax = load_alexa('data/domain/top-1000.csv')
        cry = load_dga('data/domain/dga-cryptolocke-1000.txt')
        goz = load_dga('data/domain/dga-post-tovar-goz-1000.txt')

        x = np.concatenate((aleax, cry, goz))
        y = np.concatenate(([0] * len(aleax), [1] * len(cry), [2] * len(goz)))

        return x, y

    @staticmethod
    def load_mnist():
        """加载MNIST数据集，MNIST是一个入门级的计算机视觉数据集，它包含各种手写数字图片，也包含每一张图片对应的标签，告诉我们这个是数字几"""
        with gzip.open('data/MNIST/mnist.pkl.gz', "rb") as fp:
            training_data, valid_data, test_data = pickle.load(fp, encoding="bytes")

        return training_data, valid_data, test_data

    @staticmethod
    def load_xss():
        """加载XSS数据集"""
        x1, y1 = load_filename('data/XSS/xss-200000.txt', 1)
        x2, y2 = load_filename('data/XSS/good-200000.txt', 0)

        return x1 + x2, y1 + y2

    @staticmethod
    def load_secrepo():
        """加载secrepo估计数据(ip/域名)"""
        ip_list = {}
        with open("data/etl-ip-domain-train.txt") as f:
            for line in f:
                (ip, domain) = line.split("\t")
                if not ip == "0.0.0.0":
                    if ip not in ip_list:
                        ip_list[ip] = {}

                    ip_list[ip][domain] = 1
        return ip_list

    @staticmethod
    def load_spambase():
        """SpamBase的数据不是原始的邮件内容而是已经特征化的数据，对应的特征是统计的关键字以及特殊符号的词频.
        一共58个属性，其中最后一个是垃圾邮件的标记位"""
        my_name = ["x%s" % i for i in range(1, 58)]
        data = pd.read_csv("data/spambase/spambase.data", header=None, names=my_name)
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        return x, y

    @staticmethod
    def load_movie_review():
        """Movie Review Data数据集包含1000条正面 的评论和1000条负面评论，被广泛应用于文本分类尤其是恶意评论识别方面。"""
        x1, y1 = load_files_lable("data/movie-review-data/review_polarity/txt_sentoken/pos/", 0)
        x2, y2 = load_files_lable("data/movie-review-data/review_polarity/txt_sentoken/neg/", 1)

        return x1 + x2, y1 + y2

    @staticmethod
    def load_us_cities(maxlen):
        """加载us城市

        :param maxlen: 序列的最大长度
        :return:
        """
        path = "data/us_cities/US_Cities.txt"
        file_lines = open(path, "r").read()
        x, y, char_idx = string_to_semi_redundant_sequences(file_lines, seq_maxlen=maxlen, redun_step=3)

        return x, y, char_idx, file_lines

    @staticmethod
    def load_wvs_password(maxlen):
        """加载us城市

        :param maxlen: 序列的最大长度
        :return:
        """
        path = "data/wvs-pass/wvs-pass.txt"
        file_lines = open(path, "r").read()
        x, y, char_idx = string_to_semi_redundant_sequences(file_lines, seq_maxlen=maxlen, redun_step=3)

        return x, y, char_idx, file_lines

    @staticmethod
    def load_enron1():
        """加载垃圾邮件"""

        x1, y1 = load_files_lable("data/enron1/ham/", 0)
        x2, y2 = load_files_lable("data/enron1/spam/", 1)

        return x1 + x2, y1 + y2
