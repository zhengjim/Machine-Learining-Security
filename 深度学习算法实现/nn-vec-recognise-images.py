from model.neural_network_vec import Network
import tensorflow as tf
import joblib


def load_dataset():
    """ 导入mnist数据 """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 扁平为1维
    x_train = x_train.reshape(x_train.shape[0], -1, 1)
    x_test = x_test.reshape(x_test.shape[0], -1, 1)

    # # 归一化
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # 转onehot
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # 扁平为1维
    y_train = y_train.reshape(y_train.shape[0], -1, 1)
    y_test = y_test.reshape(y_test.shape[0], -1, 1)

    return (x_train, y_train), (x_test, y_test)


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
        # print("预测值：%d, 实际值：%d " % (predict, label))
        if label != predict:
            error += 1
    return 1. - float(error) / float(total)


def main():
    (x_train, y_train), (x_test, y_test) = load_dataset()
    y_train = y_train.reshape(y_train.shape[0], -1, 1)

    nn = Network([784, 300, 10])
    nn.fix(x_train, y_train, 0.5, 2)

    # 检查梯度
    nn.gradient_check(x_train[0], y_train[0])
    # 保存模型
    joblib.dump(nn, 'export/nn-vec-reconginise-images-train.pkl')
    # 查看正确率
    print(evaluate(nn, x_test, y_test))  # 训练2轮的准确率：0.9704


if __name__ == '__main__':
    main()
