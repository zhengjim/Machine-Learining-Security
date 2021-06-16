from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


def main():
    mnist_data = fetch_openml("mnist_784")
    x, y = mnist_data.data / 255., mnist_data.target
    x_train, x_test = x[:60000], x[60000:]
    y_train, y_test = y[:60000], y[60000:]

    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    # hidden_layer_sizes：第i个元素表示第i个隐藏层中的神经元数量。
    # slover：{‘lbfgs’，‘sgd’，‘adam’}，默认’adam’。权重优化的求解器：'lbfgs’是准牛顿方法族的优化器；'sgd’指的是随机梯度下降。'adam’是指由Kingma
    # alpha：L2惩罚（正则化项）参数
    # random_state：默认无随机数生成器的状态或种子
    # max_iter：最大迭代次数
    # verbose：是否将进度消息打印到stdout
    # 优化的容忍度，容差优化
    # learning_rate_init：初始学习率

    mlp.fit(x_train, y_train)

    print(mlp.score(x_train, y_train))  # 0.9868
    print(mlp.score(x_test, y_test))  # 0.97

    fig, axes = plt.subplots(4, 4)
    # 使用“全局最小值/最大值”确保以相同的比例显示所有权重
    vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
    for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())

    plt.show()


if __name__ == "__main__":
    main()
