import numpy as np
from model import IdentityActivator
from model.recursive import RecursiveLayer, TreeNode


def data_set():
    """ 初始化数据 """
    children = [
        TreeNode(np.array([[1], [2]])),
        TreeNode(np.array([[3], [4]])),
        TreeNode(np.array([[5], [6]]))
    ]
    d = np.array([[0.5], [0.8]])

    return children, d


def gradient_check():
    """ 梯度检查 """
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()

    rnn = RecursiveLayer(2, 2, IdentityActivator(), 1e-3)

    # 计算forward值
    x, d = data_set()
    rnn.forward(x[0], x[1])
    rnn.forward(rnn.root, x[2])

    # 求取sensitivity map
    sensitivity_array = np.ones((rnn.node_width, 1),
                                dtype=np.float64)
    # 计算梯度
    rnn.backward(sensitivity_array)

    # 检查梯度
    epsilon = 10e-4
    for i in range(rnn.W.shape[0]):
        for j in range(rnn.W.shape[1]):
            rnn.W[i, j] += epsilon
            rnn.reset_state()
            rnn.forward(x[0], x[1])
            rnn.forward(rnn.root, x[2])
            err1 = error_function(rnn.root.data)
            rnn.W[i, j] -= 2 * epsilon
            rnn.reset_state()
            rnn.forward(x[0], x[1])
            rnn.forward(rnn.root, x[2])
            err2 = error_function(rnn.root.data)
            expect_grad = (err1 - err2) / (2 * epsilon)
            rnn.W[i, j] += epsilon
            print('weights(%d,%d): expected - actural %.4e - %.4e' % (i, j, expect_grad, rnn.W_grad[i, j]))

    return rnn


def test():
    children, d = data_set()
    rnn = RecursiveLayer(2, 2, IdentityActivator(), 1e-3)
    rnn.forward(children[0], children[1])
    rnn.dump()
    rnn.forward(rnn.root, children[2])
    rnn.dump()
    rnn.backward(d)
    rnn.dump(dump_grad='true')
    return rnn


if __name__ == '__main__':
    # gradient_check()
    test()
