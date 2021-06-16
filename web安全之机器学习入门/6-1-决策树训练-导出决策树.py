from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus


def main():
    iris = load_iris()
    x = iris.data
    y = iris.target

    dec = DecisionTreeClassifier()
    dec.fit(x, y)

    # 导出决策树
    tree_dot = export_graphviz(dec, out_file=None)

    # 可视化 转pdf
    graph = pydotplus.graph_from_dot_data(tree_dot)
    graph.write_pdf('export/6-1-iris.pdf')


if __name__ == "__main__":
    main()
