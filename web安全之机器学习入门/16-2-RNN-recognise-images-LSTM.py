import tensorflow as tf
from sklearn.model_selection import train_test_split


def main():
    # 导入数据
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # 转onehot
    y_train_onehot = tf.keras.utils.to_categorical(y_train)
    y_test_onehot = tf.keras.utils.to_categorical(y_test)

    # 归一化
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 顺序模型（层直接写在里面，省写add）
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),  # 输入层 28 * 28的向量
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128),
        # tf.keras.layers.Dropout(0.5),  # 丢弃50%，防止过拟合
        tf.keras.layers.Dense(10, activation="softmax"),  # 输出层 有10个类，用 softmax 概率分布
    ])

    # 编译模型
    model.compile(
        optimizer="adam",  # 优化器
        loss="categorical_crossentropy",  # 损失函数
        metrics=["acc"],  # 观察值， acc正确率
    )

    # 训练
    model.fit(
        x_train, y_train_onehot,
        batch_size=32,  # 一次放入多少样本
        epochs=10,
        validation_data=(x_test, y_test_onehot),
    )
    # loss: 0.2157 - acc: 0.9183 - val_loss: 0.2799 - val_acc: 0.8984


if __name__ == "__main__":
    main()
