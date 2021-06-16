from datasets import Datasets
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 特征提取，使用词集将操作命令向量化，根据操作统计命令词集来判断
def get_feature(cmd, fdist):
    feature = []
    for block in cmd:
        v = [0] * len(fdist)
        for i in range(0, len(fdist)):
            if fdist[i] in block:
                v[i] += 1
        feature.append(v)
    return feature


def main():
    # 导入数据
    data, y, fdist = Datasets.load_Schonlau('User3')
    x = get_feature(data, fdist)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

    num_words = len(x)
    # 序列编码one-hot
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

    # 顺序模型（层直接写在里面，省写add）
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=num_words + 1,  # 字典长度 加1 不然会报错
            output_dim=128,
            input_length=100,  # 当输入序列的长度固定时，该值为其长度
        ),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])

    # 编译模型
    model.compile(
        optimizer="adam",  # 优化器
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
    # loss: 0.2557 - acc: 0.9143 - val_loss: 0.1812 - val_acc: 0.9556

    # 画图 正确率(是否过拟合)
    plt.plot(history.epoch, history.history.get("acc"))
    plt.plot(history.epoch, history.history.get("val_acc"))
    plt.show()


if __name__ == "__main__":
    main()
