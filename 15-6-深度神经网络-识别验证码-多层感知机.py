from datasets import Datasets
from exts import get_one_hot
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# 多层感知机
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

    # x 输入28 * 28 向量， y 维度为10的向量
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    # drop_out的比例
    keep_prob = tf.placeholder(tf.float32)

    # 定义整个系统中的变量W1、b1、W2、b2 。隐藏层具有300个节点
    h1_units = 300
    W1 = tf.Variable(tf.truncated_normal([784, h1_units], stddev=0.1))
    b1 = tf.Variable(tf.zeros([h1_units]))
    W2 = tf.Variable(tf.zeros([h1_units, 10]))
    b2 = tf.Variable(tf.zeros([10]))

    # 隐藏层有一层(relu函数)
    hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

    # 定义衰减函数，这里的衰减函数使用交叉熵来衡量，通过Adagrad自适应调节，学习速率为0.3(训练模型)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

    # 初始化全部变量并定义会话
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # 顺序取出100个数据用于训练
    for i in range(int(len(x_train) / batch_size)):
        batch_xs = x_train[(i * batch_size):((i + 1) * batch_size)]
        batch_ys = y_train[(i * batch_size):((i + 1) * batch_size)]
        # 整个训练的次数取决于整个数据集合的长度以及每次训练的数据个数，其中keep_prob比例为75%
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.75})

    # 验证模型
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test, keep_prob: 1.0}))  # 0.9519


if __name__ == "__main__":
    main()
