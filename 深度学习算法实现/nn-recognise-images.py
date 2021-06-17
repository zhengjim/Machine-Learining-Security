from model.neural_network import Network
import tensorflow as tf
import joblib


def image2vec(picture):
    """ 将图像转化为样本的输入向量 """
    sample = []
    for i in range(3):
        for j in range(1):
            sample.append(picture[i][j])
    return sample


def load_dataset():
    """ 导入mnist数据 """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 转为4D tensor,MNIST是灰度的，所以我们只有一个通道
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # # 归一化
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # 转onehot
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    return (x_train.tolist(), y_train.tolist()), (x_test.tolist(), y_test.tolist())


def get_result(vec):
    """ 手写数字识别 网络的输出是一个多维向量，这个向量第n个(从0开始编号)元素的值最大，那么n就是网络的识别结果 """
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index


def evaluate(network, test_data_set, test_labels):
    """ 使用正确率评估模型，比较直观 """
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return 1. - float(error) / float(total)


def main():
    x_train = [
        [[0.], [0.], [0.]],
        [[1.], [1.], [1.]],
        [[2.], [2.], [2.]],
        [[3.], [3.], [3.]]
    ]
    y_train = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    # (x_train, y_train), (x_test, y_test) = load_dataset()
    # x_train = x_train[:10]
    # y_train = y_train[:10]

    x_train = [image2vec(i) for i in x_train]

    nn = Network([3, 2, 4])
    nn.fix(x_train, y_train, 0.5, 10)
    joblib.dump(nn, "export/nn-train.pkl")
    print(evaluate(nn, x_train, y_train))

    print(get_result(nn.predict([1., 1., 1.])))
    print(get_result(nn.predict([2., 2., 2.])))
    print(get_result(nn.predict([3., 3., 3.])))


if __name__ == '__main__':
    main()
