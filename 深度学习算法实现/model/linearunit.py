from model.perceptron import Perceptron


# 除了激活函数不同之外，两者的模型和训练规则是一样的。
# 继承感知器，激活函数换成线性
class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''初始化线性单元，设置输入参数的个数'''
        Perceptron.__init__(self, input_num, lambda x: x)
