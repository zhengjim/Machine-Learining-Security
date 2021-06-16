from functools import reduce


class Perceptron(object):
    def __init__(self, input_num, activator):
        ''' 初始化感知器，

        :param input_num: 设置输入参数的个数
        :param activator: 激活函数
        '''
        self.activator = activator
        # 初始化权重
        self.w = [0.0 for _ in range(input_num)]
        # 初始化偏置项
        self.b = 0.0

    def __str__(self):
        '''打印学习到的权重、偏置项

        :return:
        '''

        return 'w\t:%s\nb\t:%f\n' % (self.w, self.b)

    def fix(self, x, y, iteration_num, rate):
        """训练数据

        :param x: 输入训练数据，一组向量
        :param y: 对应的标签
        :param iteration_num: 训练轮数
        :param rate: 学习速率
        """
        for i in range(iteration_num):
            self._one_tarin(x, y, rate)

    def _one_tarin(self, x, y, rate):
        """训练所有的数据(一次)

        :param x: 输入训练数据，一组向量
        :param y: 对应的标签
        :param rate: 学习速率
        :return:
        """
        for (x, y) in zip(x, y):
            # 预测结果
            t = self.predict(x)
            # 更新权重,求预测值与实际值的差距
            gap = y - t
            # 利用公式求w，b。 w = wi + r(t-y)xi  b = b + r(t-y)
            self.w = list(map(lambda x_w: x_w[1] + rate * gap * x_w[0], zip(x, self.w)))
            self.b = self.b + gap * rate

    def predict(self, x):
        """输入x预测结果 使用了SGD(随机梯度下降算法)

        :param x: 向量
        :return:
        """

        # 返回[x1*w1, x2*w2, x3*w3...]
        sample = map(lambda x_w: x_w[0] * x_w[1], zip(x, self.w))
        # 计算 0.0 + x1*w1 + x2*w2 + x3*w3 +...  初始值为0.0
        y = reduce(lambda a, b: a + b, sample, 0.0)
        # 加上偏置项
        y = y + self.b
        # 激活函数
        y = self.activator(y)

        return y
