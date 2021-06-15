from datasets import Datasets
from exts import get_one_hot
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# softmax回归算法
def main():
    # 加载数据集
    train_data, vaild_data, test_data = Datasets.load_mnist()
    x_train, y_train = train_data
    x_test, y_test = test_data

    # one-hot编码又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都有其独立的寄存器位，并且在任意时刻，其中只有一位有效。
    y_train = get_one_hot(y_train)
    y_test = get_one_hot(y_test)

    # 每次训练的数据子集的个数
    batch_size = 100

    # 设置占位符 x对应整个系统输入，是一个维度为784的向量集合，且长度不限制
    x = tf.placeholder("float", [None, 784])
    # y_整个系统的输出，对 应一个维度为10的向量集合
    y_ = tf.placeholder("float", [None, 10])

    # x的维度为784，所以W为一个784乘以10的数组
    W = tf.Variable(tf.zeros([784, 10]))
    # b是一个维度为10的向量
    b = tf.Variable(tf.zeros([10]))

    # 整个系统的操作函数(softmax回归算法)
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # 定义衰减函数，这里的衰减函数使用交叉熵来衡量，通过梯度下降算法以0.01的学习速率最小化交叉熵(训练模型)
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # 初始化全部变量并定义会话
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # 顺序取出100个数据用于训练
    for i in range(int(len(x_train) / batch_size)):
        batch_xs = x_train[(i * batch_size):((i + 1) * batch_size)]
        batch_ys = y_train[(i * batch_size):((i + 1) * batch_size)]

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 数据验证(评估我们的模型)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 把布尔值转换成浮点数，然后取平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))  # 0.9097


if __name__ == "__main__":
    main()
