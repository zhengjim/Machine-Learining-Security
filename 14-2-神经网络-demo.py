from sklearn.neural_network import MLPClassifier


def main():
    x = [[0., 0, ], [1., 1.]]
    y = [0, 1]

    # 隐藏层一共两层,对应神经元个数分别为5个和2个
    # hidden_layer_sizes：第i个元素表示第i个隐藏层中的神经元数量。
    # slover：{‘lbfgs’，‘sgd’，‘adam’}，默认’adam’。权重优化的求解器：'lbfgs’是准牛顿方法族的优化器；'sgd’指的是随机梯度下降。'adam’是指由Kingma
    # alpha：L2惩罚（正则化项）参数
    # random_state：默认无随机数生成器的状态或种子
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    mlp.fit(x, y)

    print(mlp.predict([[2., 2.], [-1., -2.]]))


if __name__ == "__main__":
    main()
