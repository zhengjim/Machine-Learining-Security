import numpy as np
from model import SigmoidActivator


class FullConnectedLayer(object):
    """ 全连接层实现类 """

    def __init__(self, input_size, output_size, activator):
        """初始化

        :param input_size: 本层输入向量的维度
        :param output_size: 本层输出向量的维度
        :param activator: 激活函数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        self.W = np.random.uniform(-.1, .1, (output_size, input_size))  # 权重数组W
        self.b = np.zeros((output_size, 1))  # 偏置项b
        self.output = np.zeros((output_size, 1))  # 输出向量

    def forward(self, input_array):
        """ 前向计算
        输出向量a = sigmoid(W * 输入向量x)  ,其中 偏置项b看为1 * w

        :param input_array: 输入向量，维度必须等于input_size

        :return:
        """
        self.input = input_array

        self.output = self.activator.forward(np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        """ 反向计算W和b的梯度
        l层误差项 = l层输出向量a * (1 - l层输出向量a) * 矩阵W转置 * (l的上层误差项)

        :param delta_array: 从上一层传递过来的误差项

        :return:
        """

        self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)  # w的梯度
        self.b_grad = delta_array  # b的梯度

    def update(self, rate):
        """使用梯度下降更新权重

        :param rate: 速率

        :return:
        """
        self.W += rate * self.W_grad
        self.b += rate * self.b_grad


class Network(object):
    """ 神经网络类 """

    def __init__(self, layers):
        """ 初始化 """
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullConnectedLayer(layers[i], layers[i + 1], SigmoidActivator()))

    def predict(self, x):
        """ 预测样本

        :param x: 样本
        :return:
        """
        output = x
        # 输出等于下层的输入
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def fix(self, x, y, rate, epoch):
        """ 训练模型

        :param x: 样本
        :param y: 样本标签
        :param rate: 速率
        :param epoch: 训练轮数

        :return:
        """
        for _ in range(epoch):
            for i in range(len(x)):
                self.__train_one_sample(x[i], y[i], rate)

    def __train_one_sample(self, x, y, rate):
        """ 用一个样本训练网络

        :param x: 样本
        :param y: 样本标签
        :param rate: 速率
        :return:
        """
        self.predict(x)
        self.__calc_delta(y)
        self.__update_W(rate)

    def __calc_delta(self, y):
        """计算每层误差项
        输出层误差项公式： y * (1-y) * (t-y)  y是输出值 t是实际值

        :param y: 样本标签
        :return:
        """
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (y - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta

    def __update_W(self, rate):
        """ 更新权重矩阵

        :param rate: 速率
        :return:
        """
        for layer in self.layers:
            layer.update(rate)

    def gradient_check(self, x, y):
        """梯度检查

        :param x: 样本的特征
        :param y: 样本的标签

        :return:
        """

        # 获取网络在当前样本下每个连接的梯度
        self.predict(x)
        self.__calc_delta(y)

        # 检查梯度
        epsilon = 10e-4
        for layer in self.layers:
            for i in range(layer.W.shape[0]):
                for j in range(layer.W.shape[1]):
                    # layer.W[i, j] 为获取的指定梯度
                    # 加一个很小的值，计算网络的误差
                    layer.W[i, j] += epsilon
                    output = self.predict(x)
                    err1 = self.__loss(y, output)
                    # 减去一个很小的值，计算网络的误差
                    layer.W[i, j] -= 2 * epsilon  # 刚才加过了一次，因此这里需要减去2倍
                    output = self.predict(x)
                    err2 = self.__loss(y, output)
                    # 算期望的梯度值
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    # 刚减了，所以要加回去等于原来梯度
                    layer.W[i, j] += epsilon
                    print('W(%d,%d): expected: %.4e , actural: %.4e' % (i, j, expect_grad, layer.W_grad[i, j]))

    def __loss(self, output, label):
        """损失值 平方差

        :param output: 预测值
        :param label:  实际值

        :return:
        """

        return 0.5 * ((label - output) * (label - output)).sum()
