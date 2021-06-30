import numpy as np


def padding(input_array, zp):
    """为数组增加Zero padding，自动适配输入为2D和3D的情况


    :param input_array: 输入数组
    :param zp:
    :return: 填充几圈
    """

    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((input_depth, input_height + 2 * zp, input_width + 2 * zp))
            padded_array[:, zp: zp + input_height, zp: zp + input_width] = input_array

            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((input_height + 2 * zp, input_width + 2 * zp))
            padded_array[zp: zp + input_height, zp: zp + input_width] = input_array

            return padded_array


def conv(input_array, kernel_array, output_array, stride, bias):
    """计算卷积，自动适配输入为2D和3D的情况

    :param input_array: 输入数组
    :param kernel_array: 权重数组
    :param output_array: 输出数组
    :param stride: 步长
    :param bias: 偏置值

    :return:
    """

    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            # 获取卷积的区域
            conv_value = get_patch(input_array, i, j, kernel_width, kernel_height, stride)
            output_array[i][j] = (conv_value * kernel_array).sum() + bias


def get_patch(input_array, i, j, filter_width, filter_height, stride):
    """ 从输入数组中获取本次卷积的区域，自动适配输入为2D和3D的情况

    :param input_array: 输入数组
    :param i: 输出高度
    :param j: 输出宽度
    :param filter_width: filter宽度
    :param filter_height: filter高度
    :param stride: 步长
    :return:
    """
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        return input_array[start_i: start_i + filter_height, start_j: start_j + filter_width]
    elif input_array.ndim == 3:
        return input_array[:, start_i: start_i + filter_height, start_j: start_j + filter_width]


def get_max_index(array):
    """获取一个2D区域的最大值所在的索引

    :param array: 区域数组
    :return:
    """

    max_i = 0
    max_j = 0
    max_value = array[0, 0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > max_value:
                max_value = array[i, j]
                max_i, max_j = i, j
    return max_i, max_j


def element_wise_op(array, op):
    """对numpy数组进行element wise操作(按元素操作)

    :param array: np.array数组
    :param op: 激活函数

    :return:
    """

    for i in np.nditer(array, op_flags=['readwrite']):  # 修改数组值 readwrite
        i[...] = op(i)


class ConvLayer(object):
    """ 卷积层的实现 """

    def __init__(self, input_width, input_height, channel_number, filter_width, filter_height, filter_number,
                 zero_padding, stride, activator, learning_rate):
        """设置超参数，初始化卷积层

        :param input_width: 输入层宽度
        :param input_height: 输入层高度
        :param channel_number: 通道个数
        :param filter_width: filter宽度
        :param filter_height: filter高度
        :param filter_number: filter个数
        :param zero_padding: 在原图像外围填充多少圈的0
        :param stride: 步长
        :param activator: 激活函数
        :param learning_rate: 学习速率
        """

        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.activator = activator
        self.learning_rate = learning_rate
        # 输出宽度
        self.output_width = ConvLayer.calculate_output_size(self.input_width, filter_width, zero_padding, stride)
        # 输出高度
        self.output_height = ConvLayer.calculate_output_size(self.input_height, filter_height, zero_padding, stride)
        # 卷积层的结果输出
        self.output_array = np.zeros((self.filter_number, self.output_height, self.output_width))
        # 输出filter
        self.filters = [Filter(filter_width, filter_height, self.channel_number) for _ in range(filter_number)]

    def forward(self, input_array):
        """计算根据输入来计算卷积层的输出，输出结果保存在self.output_array

        :param input_array: 输入
        :return:
        """

        self.input_array = input_array
        # 为数组外围填充0
        self.padded_input_array = padding(input_array, self.zero_padding)
        for i in range(self.filter_number):
            filter_ = self.filters[i]
            conv(self.padded_input_array, filter_.get_weights(), self.output_array[i], self.stride, filter_.get_bias())
        # output_array每个元素激活函数
        element_wise_op(self.output_array, self.activator.forward)

    def backward(self, input_array, sensitivity_array, activator):
        """计算传递给前一层的误差项，以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array
        梯度保存在Filter对象的weights_grad

        :param input_array:
        :param sensitivity_array:
        :param activator:

        :return:
        """
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array, activator)
        self.bp_gradient(sensitivity_array)

    def bp_sensitivity_map(self, sensitivity_array, activator):
        """计算传递到上一层的sensitivity map

        :param sensitivity_array: 本层的sensitivity map
        :param activator: 上一层的激活函数

        :return:
        """
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        # full卷积，对误差矩阵进行zero padding
        # 虽然原始输入的zero padding单元也会获得残差
        # 但这个残差不需要继续向上传递，因此就不计算了
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width + self.filter_width - 1 - expanded_width) // 2
        padded_array = padding(expanded_array, zp)
        # 初始化delta_array，用于保存传递到上一层的sensitivity map
        self.delta_array = self.create_delta_array()
        # 对于具有多个filter的卷积层来说，最终传递到上一层的sensitivity map相当于所有的filter的sensitivity map之和
        for f in range(self.filter_number):
            filter_ = self.filters[f]
            # 将filter权重翻转180度
            flipped_weights = np.array(list(map(lambda i: np.rot90(i, 2), filter_.get_weights())))
            # 计算与一个filter对应的delta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d], delta_array[d], 1, 0)
            self.delta_array += delta_array
        # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, activator.backward)
        self.delta_array *= derivative_array

    def create_delta_array(self):
        """初始化误差矩阵"""
        return np.zeros((self.channel_number, self.input_height, self.input_width))

    def expand_sensitivity_map(self, sensitivity_array):
        """将步长为S的sensitivity map "还原"为步长为1的sensitivity map

        :param sensitivity_array: sensitivity map

        :return:
        """
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小,计算stride为1时sensitivity map的大小
        expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height - self.filter_height + 2 * self.zero_padding + 1)
        # 构建新的sensitivity_map
        expand_array = np.zeros((depth, expanded_height, expanded_width))
        # 从原始sensitivity map拷贝误差值
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = sensitivity_array[:, i, j]
        return expand_array

    def bp_gradient(self, sensitivity_array):
        """计算梯度

        :param sensitivity_array: sensitivity map

        :return:
        """
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_number):
            # 计算每个权重的梯度
            filter_ = self.filters[f]
            for d in range(filter_.weights.shape[0]):
                conv(self.padded_input_array[d], expanded_array[f], filter_.weights_grad[d], 1, 0)
            # 计算偏置项的梯度
            filter_.bias_grad = expanded_array[f].sum()

    def update(self):
        """按照梯度下降，更新权重

        :return:
        """
        for filter_ in self.filters:
            filter_.update(self.learning_rate)

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        """ 计算输出数组大小

        :param input_size:  对应长或宽输入大小
        :param filter_size: 对应filter长或宽输入大小
        :param zero_padding: zero_padding 圈数
        :param stride: 步长
        :return:
        """

        return (input_size - filter_size + 2 * zero_padding) // stride + 1


class Filter(object):
    """ Filter类保存了卷积层的参数以及梯度，并且实现了用梯度下降算法来更新参数。 """

    def __init__(self, width, height, depth):
        """

        :param width: 宽度
        :param height: 长度
        :param depth: 通道数
        """
        # 权重
        self.weights = np.random.uniform(-1e-4, 1e-4, (depth, height, width))
        # 偏置值
        self.bias = 0
        # 权重梯度
        self.weights_grad = np.zeros(self.weights.shape)
        # 偏置值梯度
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        """ 梯度下降更新权重和偏置值 """
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad


class MaxPoolingLayer(object):
    """ Pool层实现 """

    def __init__(self, input_width, input_height, channel_number, filter_width, filter_height, stride):
        """初始化Pool层

        :param input_width: 输入宽度
        :param input_height: 输入高度
        :param channel_number: 通道个数
        :param filter_width: filter宽度
        :param filter_height:  filter高度
        :param stride: 步长
        """
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        # 输出宽
        self.output_width = (input_width - filter_width) // self.stride + 1
        # 输出高
        self.output_height = (input_height - filter_height) // self.stride + 1
        # 输出数组
        self.output_array = np.zeros((self.channel_number, self.output_height, self.output_width))

    def forward(self, input_array):
        """根据输入来输出池化(pool)层的结果

        :param input_array:  输入

        :return:
        """
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = (
                        get_patch(input_array[d], i, j, self.filter_width, self.filter_height, self.stride).max())

    def backward(self, input_array, sensitivity_array):
        """计算传递给前一层的误差项，以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array

        :param input_array: 输入
        :param sensitivity_array: sensitivity map
        :return:
        """
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(input_array[d], i, j, self.filter_width, self.filter_height, self.stride)
                    k, l = get_max_index(patch_array)
                    self.delta_array[d, i * self.stride + k, j * self.stride + l] = sensitivity_array[d, i, j]
