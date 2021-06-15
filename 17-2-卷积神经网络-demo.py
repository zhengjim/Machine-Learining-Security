import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    # 导入数据
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

    # 顺序模型（层直接写在里面，省写add）
    model = tf.keras.Sequential([
        # 第一个卷积层
        tf.keras.layers.Conv2D(
            filters=32,  # 32个卷积核
            kernel_size=(3, 3),  # 大小3*3
            padding='Same',  # 卷积模式
            activation="relu",  # 激活函数
            input_shape=(28, 28, 1),  # 输入张量的大小
        ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),  # 池化
        tf.keras.layers.Dropout(.25),  # 丢弃25% 防止过拟合
        # 图像需要扁平成一维的 用Flatten层
        tf.keras.layers.Flatten(),
        # 全连接层
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(.25),
        # 输出层 有10个类
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    # 编译模型
    sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=sgd,  # 优化器
        loss="categorical_crossentropy",  # 损失函数
        metrics=["acc"],  # 观察值， acc正确率
    )

    # 训练
    history = model.fit(
        x_train, y_train,
        batch_size=32,  # 一次放入多少样本
        epochs=10,
        validation_data=(x_test, y_test),
    )
    # loss: 0.0211 - acc: 0.9933 - val_loss: 0.0334 - val_acc: 0.9886

    # 画图 正确率(是否过拟合)
    plt.plot(history.epoch, history.history.get("acc"))
    plt.plot(history.epoch, history.history.get("val_acc"))
    plt.show()


if __name__ == "__main__":
    main()
