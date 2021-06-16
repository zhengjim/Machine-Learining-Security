from datasets import Datasets
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


def main():
    # 导入数据
    x, y = Datasets.load_movie_review()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

    # 数据预处理，词袋
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(x)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    # 序列编码one-hot
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)

    print(gnb.score(x_test, y_test))  # 0.5766666666666667


if __name__ == "__main__":
    main()
