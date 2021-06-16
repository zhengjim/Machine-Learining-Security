from sklearn.neural_network import MLPClassifier
from datasets import Datasets
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer


def main():
    # 加载ADFA-LD 数据
    x1, y1 = Datasets.load_adfa_normal()
    x2, y2 = Datasets.load_adfa_attack(r"Java_Meterpreter_\d+/UAD-Java-Meterpreter*")
    x = x1 + x2
    y = y1 + y2

    # 词袋特征
    cv = CountVectorizer(min_df=1)
    x = cv.fit_transform(x).toarray()

    mlp = MLPClassifier(hidden_layer_sizes=(150, 50), max_iter=10000, alpha=1e-4, solver='sgd', tol=1e-4,
                        random_state=1, learning_rate_init=.01)
    # hidden_layer_sizes：第i个元素表示第i个隐藏层中的神经元数量。
    # slover：{‘lbfgs’，‘sgd’，‘adam’}，默认’adam’。权重优化的求解器：'lbfgs’是准牛顿方法族的优化器；'sgd’指的是随机梯度下降。'adam’是指由Kingma
    # alpha：L2惩罚（正则化项）参数
    # random_state：默认无随机数生成器的状态或种子
    # max_iter：最大迭代次数
    # verbose：是否将进度消息打印到stdout
    # 优化的容忍度，容差优化
    # learning_rate_init：初始学习率

    scores = cross_val_score(mlp, x, y, cv=10, scoring="accuracy")
    print(scores.mean())  # 0.9654934210526316


if __name__ == "__main__":
    main()
