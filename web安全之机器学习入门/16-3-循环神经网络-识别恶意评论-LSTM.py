from datasets import Datasets
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    # 导入数据
    x, y = Datasets.load_movie_review()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

    # 数据预处理，词袋
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(x)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    num_words = len(tokenizer.word_index)
    # 序列编码one-hot
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

    # 顺序模型（层直接写在里面，省写add）
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=num_words + 1,  # 字典长度 加1 不然会报错
            output_dim=300,  # 全连接嵌入的维度，常用256或300
            input_length=200,  # 当输入序列的长度固定时，该值为其长度
            trainable=True,  # 代表词向量作为参数进行更新
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, implementation=2)),  # 双向LSTM
        tf.keras.layers.Dropout(0.5),  # 丢弃50%，防止过拟合
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
    # loss: 0.0013 - acc: 1.0000 - val_loss: 0.5421 - val_acc: 0.7350

    # 画图 正确率(是否过拟合)
    plt.plot(history.epoch, history.history.get("acc"))
    plt.plot(history.epoch, history.history.get("val_acc"))
    plt.show()


if __name__ == "__main__":
    main()
