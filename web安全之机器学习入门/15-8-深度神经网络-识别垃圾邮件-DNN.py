from datasets import Datasets
import tensorflow as tf
from sklearn.model_selection import train_test_split


def input_fn(features, labels, training=True, batch_size=32):
    '''用于训练或评估的输入函数

    :param features: 特征
    :param labels: 标签
    :param training: 是否是训练
    :param batch_size: 一次放入多少样本
    :return:
    '''
    # 将输入转换为数据集
    dataset = tf.compat.v1.data.Dataset.from_tensor_slices((dict(features), labels))
    # 如果训练模式下，shuffle 打乱 、 repeat 重复
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)


# DNN
def main():
    # 加载数据集
    x, y = Datasets.load_spambase()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

    # 特征列转换为tf识别的格式
    my_feature_columns = [tf.compat.v1.feature_column.numeric_column("x%s" % i) for i in range(1, 57)]

    # DNN
    classifier = tf.compat.v1.estimator.DNNClassifier(
        feature_columns=my_feature_columns,  # 特征列
        hidden_units=[30, 10],  # 表示隐含层是30*10的神经网络
        n_classes=2,  # 输出层的分类有2个
    )

    # 训练 需要输入训练数据，并指定训练的步数。这一步需要和tf.data.Dataset结合使用。使用tf.data.Dataset进行每一个批次的数据喂取
    classifier.train(
        input_fn=lambda: input_fn(x_train, y_train, training=True),
        steps=500,
    )

    # 评估模型
    eval_result = classifier.evaluate(
        input_fn=lambda: input_fn(x_test, y_test, training=False),
    )
    print(eval_result["accuracy"])  # 0.90658945


if __name__ == "__main__":
    main()
