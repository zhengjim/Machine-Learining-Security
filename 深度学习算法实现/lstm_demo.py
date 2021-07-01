import numpy as np
from model.lstm import LstmLayer
from model import IdentityActivator


def data_set():
    x = [np.array([[1], [2], [3]]),
         np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    return x, d


def gradient_check():
    """ 梯度检查 """
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()

    lstm = LstmLayer(3, 2, 1e-3)

    # 计算forward值
    x, d = data_set()
    lstm.forward(x[0])
    lstm.forward(x[1])

    # 求取sensitivity map
    sensitivity_array = np.ones(lstm.h_list[-1].shape, dtype=np.float64)
    # 计算梯度
    lstm.backward(x[1], sensitivity_array, IdentityActivator())

    # 检查梯度
    epsilon = 10e-4
    for i in range(lstm.Wfh.shape[0]):
        for j in range(lstm.Wfh.shape[1]):
            lstm.Wfh[i, j] += epsilon
            lstm.reset_state()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err1 = error_function(lstm.h_list[-1])
            lstm.Wfh[i, j] -= 2 * epsilon
            lstm.reset_state()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err2 = error_function(lstm.h_list[-1])
            expect_grad = (err1 - err2) / (2 * epsilon)
            lstm.Wfh[i, j] += epsilon
            print('weights(%d,%d): expected - actural %.4e - %.4e' % (i, j, expect_grad, lstm.Wfh_grad[i, j]))
    return lstm


def test():
    l = LstmLayer(3, 2, 1e-3)
    x, d = data_set()
    l.forward(x[0])
    l.forward(x[1])
    l.backward(x[1], d, IdentityActivator())
    return l


if __name__ == '__main__':
    gradient_check()
    # test()
