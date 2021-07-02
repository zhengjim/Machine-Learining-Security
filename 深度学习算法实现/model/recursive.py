import numpy as np
from model import IdentityActivator


class TreeNode(object):
    """ 树节点结构,用它保存卷积神经网络生成的整棵树 """

    def __init__(self, data, children=None, children_data=None):
        """初始化树节点结构

        :param data:  输入数据
        :param children:  子节点
        :param children_data:  父节点
        """
        if children_data is None:
            children_data = []
        if children is None:
            children = []
        self.parent = None
        self.children = children
        self.children_data = children_data
        self.data = data
        for child in children:
            child.parent = self


class RecursiveLayer(object):
    """ 递归神经网络实现 """

    def __init__(self, node_width, child_count, activator, learning_rate):
        """递归神经网络构造函数

        :param node_width:  表示每个节点的向量的维度
        :param child_count: 每个父节点有几个子节点
        :param activator: 激活函数
        :param learning_rate: 学习速率
        """
        self.node_width = node_width
        self.child_count = child_count
        self.activator = activator
        self.learning_rate = learning_rate
        # 权重数组W
        self.W = np.random.uniform(-1e-4, 1e-4, (node_width, node_width * child_count))
        # 偏置项b
        self.b = np.zeros((node_width, 1))
        # 递归神经网络生成的树的根节点
        self.root = None

    def forward(self, *children):
        """前向计算
        递归神经网络将这些树节点作为子节点，并计算它们的父节点。最后，将计算的父节点保存在self.root变量中

        :param children: 一系列的树节点对象

        """
        children_data = self.concatenate(children)
        parent_data = self.activator.forward(np.dot(self.W, children_data) + self.b)
        self.root = TreeNode(parent_data, children, children_data)

    def concatenate(self, tree_nodes):
        """将各个树节点中的数据拼接成一个长向量

        :param tree_nodes: 各个树节点

        :return:
        """
        concat = np.zeros((0, 1))
        for node in tree_nodes:
            concat = np.concatenate((concat, node.data))

        return concat

    def backward(self, parent_delta):
        """BPTS反向传播算法

        :param parent_delta:  父节点误差
        :return:
        """
        # 各个节点的误差项
        self.calc_delta(parent_delta, self.root)
        # 梯度
        self.W_grad, self.b_grad = self.calc_gradient(self.root)

    def calc_delta(self, parent_delta, parent):
        """计算每个节点的delta

        :param parent_delta:  父节点误差
        :param parent:  父节点

        :return:
        """
        parent.delta = parent_delta
        if parent.children:
            # 根据式2计算每个子节点的delta
            children_delta = np.dot(self.W.T, parent_delta) * (self.activator.backward(parent.children_data))
            # slices = [(子节点编号，子节点delta起始位置，子节点delta结束位置)]
            slices = [(i, i * self.node_width, (i + 1) * self.node_width) for i in range(self.child_count)]
            # 针对每个子节点，递归调用calc_delta函数
            for s in slices:
                self.calc_delta(children_delta[s[1]:s[2]], parent.children[s[0]])

    def calc_gradient(self, parent):
        """计算每个节点权重的梯度，并将它们求和，得到最终的梯度

        :param parent: 父节点
        :return:
        """
        # 初始化
        W_grad = np.zeros((self.node_width, self.node_width * self.child_count))
        b_grad = np.zeros((self.node_width, 1))
        if not parent.children:
            return W_grad, b_grad
        parent.W_grad = np.dot(parent.delta, parent.children_data.T)
        parent.b_grad = parent.delta
        W_grad += parent.W_grad
        b_grad += parent.b_grad
        # 将每个节点梯度求和
        for child in parent.children:
            W, b = self.calc_gradient(child)
            W_grad += W
            b_grad += b
        return W_grad, b_grad

    def update(self):
        """ 使用SGD算法更新权重 """
        self.W -= self.learning_rate * self.W_grad
        self.b -= self.learning_rate * self.b_grad

    def reset_state(self):
        """ 重置父节点 """
        self.root = None

    def dump(self, dump_grad=False):
        """ 打印递归网络 """
        print('root.data: %s' % self.root.data)
        print('root.children_data: %s' % self.root.children_data)
        if dump_grad:
            print('W_grad: %s' % self.W_grad)
            print('b_grad: %s' % self.b_grad)
