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
