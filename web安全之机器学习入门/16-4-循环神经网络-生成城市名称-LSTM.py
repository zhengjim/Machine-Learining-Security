from datasets import Datasets
from exts import sample
import tensorflow as tf
import random
import numpy as np


def main():
    # 导入数据
    maxlen = 20
    x, y, char_idx, file_lines = Datasets.load_us_cities(maxlen=20)

    # 建立模型
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[maxlen, len(char_idx)]),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(char_idx), activation="softmax")
    ])

    # 编译模型
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["acc"],
    )

    char_indices = dict((c, i) for i, c in enumerate(char_idx))
    indices_char = dict((i, c) for i, c in enumerate(char_idx))

    for epoch in range(40):
        rand_index = random.randint(0, len(file_lines) - maxlen - 1)
        seed = file_lines[rand_index: rand_index + maxlen]
        # 训练
        model.fit(x, y, epochs=1, batch_size=128)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print("[*]Diversity:", diversity)
            generated = ""
            for i in range(30):
                x_pred = np.zeros((1, maxlen, len(char_idx)))
                for t, char in enumerate(seed):
                    x_pred[0, t, char_indices[char]] = 1.0
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]
                seed = seed[1:] + next_char
                generated += next_char

            print("[*]Generated: ", generated)
            print()


if __name__ == "__main__":
    main()
