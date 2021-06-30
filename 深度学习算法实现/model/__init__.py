import numpy as np


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1


class ReluActivator(object):
    """ relu激活函数 """

    def forward(self, weighted_input):
        return max(0, weighted_input)

    def backward(self, output):
        return 1 if output > 0 else 0


class SigmoidActivator(object):
    """ Sigmoid激活函数类 """

    def forward(self, weighted_input):
        """ sigmoid函数"""
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        """ sigmoid导数 """
        return output * (1 - output)


def element_wise_op(array, op):
    """对numpy数组进行element wise操作(按元素操作)

    :param array: np.array数组
    :param op: 激活函数

    :return:
    """

    for i in np.nditer(array, op_flags=['readwrite']):  # 修改数组值 readwrite
        i[...] = op(i)
