from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
import pydotplus
from datasets import Datasets


def get_guess_password_and_normal(kdd99_data):
    x = []
    y = []
    for data in kdd99_data:
        if (data[41] in ["normal.", "guess_passwd."]) and (data[2] == "pop_3"):
            # 提取结果
            y.append(1 if data[41] == "guess_passwd." else 0)
            # 提取特征
            x.append(list(map(lambda i: float(i), [data[0]] + data[4:8] + data[22:30])))
    return x, y


def main():
    kdd99_data = Datasets.load_kdd99()
    x, y = get_guess_password_and_normal(kdd99_data)

    # 决策树 交叉验证
    dec = DecisionTreeClassifier()
    scores = cross_val_score(dec, x, y, n_jobs=-1, cv=10)
    print(scores.mean())  # 0.9904371584699454

    # 导出决策树并可视化
    dec.fit(x, y)
    tree_dot = export_graphviz(dec)
    graph = pydotplus.graph_from_dot_data(tree_dot)
    graph.write_pdf('export/6-3-guess_password_tree.pdf')


if __name__ == "__main__":
    main()
