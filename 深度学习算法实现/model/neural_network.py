import random
from numpy import *
from functools import reduce

"""
Network 神经网络对象，提供API接口。它由若干层对象组成以及连接对象组成。
Layer 层对象，由多个节点组成。
Node 节点对象计算和记录节点自身的信息(比如输出值、误差项等)，以及与这个节点相关的上下游的连接。
Connection 每个连接对象都要记录该连接的权重。
Connections 仅仅作为Connection的集合对象，提供一些集合操作。
"""


class Node(object):
    def __init__(self, layer_id, node_id):
        """ 构造节点对象 """
        self.layer_id = layer_id  # 节点所属的层的编号
        self.node_id = node_id  # 节点的编号
        self.upstream = []  # 上游
        self.downstream = []  # 下游
        self.out = 0.  # 输出值
        self.delta = 0.  # 误差项

    def set_out(self, out):
        """ 设置设置输出值 (输入层用) """
        self.out = out

    def add_connection_down(self, conn):
        """ 添加一个到下游节点的连接 """
        self.downstream.append(conn)

    def add_connection_up(self, conn):
        """ 添加一个到上游节点的连接 """
        self.upstream.append(conn)

    def calc_out(self):
        """节点输出公式：y=sigmoid(w1x1+w2x2+...+b)
        其中b可以看做x=1的wx

        :return:
        """

        out = reduce(lambda ret, conn: ret + conn.node_up.out * conn.w, self.upstream, 0.)
        # sigmoid
        self.out = 1. / (1. + exp(-out))

    def calc_hidden_layer_delta(self):
        """ 计算隐藏层的误差项
        公式 a(1-a)reduce(w * 下层误差项)
        其中 a是输出值 w是权重 reduce是和式

        :return:
        """
        down_delta = reduce(lambda ret, conn: ret + conn.node_down.delta * conn.w, self.downstream, 0.0)
        self.delta = self.out * (1 - self.out) * down_delta

    def calc_out_layer_delta(self, t):
        """ 计算输出层的误差项
        公式 y(1-y)(t-y)
        其中 y是输出值 t是目标值

        :return:
        """
        self.delta = self.out * (1 - self.out) * (t - self.out)

    def __str__(self):
        """ 打印节点信息 """
        node_str = '%u-%u: out: %f delta: %f' % (self.layer_id, self.node_id, self.out, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


class ConstNode(object):
    """ 输出恒为1的节点，计算偏置项需要 """

    def __init__(self, layer_id, node_id):
        self.layer_id = layer_id  # 节点所属的层的编号
        self.node_id = node_id  # 节点的编号
        self.downstream = []  # 下游
        self.out = 1.  # 输出值恒为1

    def add_connection_down(self, conn):
        """ 添加一个到下游节点的连接 """
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        """ 计算隐藏层的误差项
        公式 a(1-a)reduce(w * 下层误差项)
        其中 a是输出值 w是权重 reduce是和式

        :return:
        """
        down_delta = reduce(lambda ret, conn: ret + conn.node_down.delta * conn.w, self.downstream, 0.)
        self.delta = self.out * (1. - self.out) * down_delta

    def __str__(self):
        """ 打印节点信息 """
        node_str = '%u-%u: output: 1' % (self.layer_id, self.node_id)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


class Layer(object):
    """ 初始化一层，提供对Node集合的操作 """

    def __init__(self, layer_id, node_num):
        self.layer_id = layer_id
        self.nodes = [Node(layer_id, i) for i in range(node_num)]
        self.nodes.append(ConstNode(layer_id, node_num))

    def set_out(self, input_shape):
        """ 设置层的输出。当层是输入层时会用到。 """
        for i in range(len(input_shape)):
            self.nodes[i].set_out(input_shape[i])

    def calc_out(self):
        """ 计算层的输出,除了输入层 """
        for node in self.nodes[:-1]:
            node.calc_out()

    def show(self):
        """ 显示层信息 """
        for node in self.nodes:
            print(node)


class Connection(object):
    """ 记录连接的权重，以及这个连接所关联的上下游节点 """

    def __init__(self, node_up, node_down):
        self.node_up = node_up  # 上游节点
        self.node_down = node_down  # 下游节点
        self.w = random.uniform(-.1, .1)  # 权重初始化一个很小的随机数
        self.gradient = .0  # 梯度

    def calc_gradient(self):
        """ 计算梯度 """
        self.gradient = self.node_up.out * self.node_down.delta

    def get_gradient(self):
        """ 获取梯度 """
        return self.gradient

    def update_w(self, rate):
        """ 梯度下降更新权重 """
        self.calc_gradient()
        self.w += rate * self.gradient

    def __str__(self):
        """ 打印连接信息 """
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.node_up.layer_id,
            self.node_up.node_id,
            self.node_down.layer_id,
            self.node_down.node_id,
            self.w)


class Connections(object):
    """ 提供Connection集合操作 """

    def __init__(self):
        self.connections = []

    def add_connection(self, conn):
        """ 添加连接 """
        self.connections.append(conn)

    def show(self):
        """ 显示连接 """
        for conn in self.connections:
            print(conn)


class Network(object):
    """ 神经网络对象，提供API接口。它由若干层对象组成以及连接对象组成。 """

    def __init__(self, layers):
        """初始化一个全连接神经网络

        :param layers: 描述神经网络每层节点数 例: [2,3,2]
        """

        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        # 添加层、节点
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        # 添加节点连接信息，连接数层数 -1
        for layer in range(layer_count - 1):
            connections = []
            # 全连接
            for node_down in self.layers[layer + 1].nodes[:-1]:
                for node_up in self.layers[layer].nodes:
                    connections.append(Connection(node_up, node_down))
            for conn in connections:
                self.connections.add_connection(conn)
                conn.node_down.add_connection_up(conn)
                conn.node_up.add_connection_down(conn)

    def fix(self, x, y, rate, iteration):
        """ 训练神经网络

        :param x: 训练样本特征。
        :param y:  训练样本标签。
        :param rate: 速率
        :param iteration: 训练次数

        :return:
        """
        for i in range(iteration):
            for d in range(len(x)):
                self.__train_one_sample(x[d], y[d], rate)

    def __train_one_sample(self, x, y, rate):
        """ 用一个样本训练网络 """
        self.predict(x)
        self.__calc_delta(y)
        self.__update_w(rate)

    def predict(self, x):
        """ 根据输入的样本预测输出值 """
        self.layers[0].set_out(x)
        # 逐层计算
        for i in range(1, len(self.layers)):
            self.layers[i].calc_out()

        return list(map(lambda node: node.out, self.layers[-1].nodes[:-1]))

    def __calc_delta(self, y):
        """ 计算每个节点的误差项 """
        # 计算输出层的误差项
        nodes_out = self.layers[-1].nodes
        for i in range(len(y)):
            nodes_out[i].calc_out_layer_delta(y[i])
        # 计算隐藏层误差项
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def __update_w(self, rate):
        """ 更新每个连接权重 """
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_w(rate)

    def __calc_gradient(self):
        """ 计算每个连接的梯度 """
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, x, y):
        """ 获得网络在一个样本下，每个连接上的梯度 """
        self.predict(x)
        self.__calc_delta(y)
        self.__calc_gradient()

    def show(self):
        """ 打印网络信息 """
        for layer in self.layers:
            layer.show()
