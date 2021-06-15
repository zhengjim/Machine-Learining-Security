from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
import pydotplus
from datasets import Datasets


def main():
    # 加载ADFA-LD 数据
    x1, y1 = Datasets.load_adfa_normal()
    x2, y2 = Datasets.load_adfa_attack(r"Hydra_FTP_\d+/UAD-Hydra-FTP*")
    x = x1 + x2
    y = y1 + y2

    # 词袋特征
    cv = CountVectorizer()
    x = cv.fit_transform(x).toarray()

    # 决策树 交叉验证
    dec = DecisionTreeClassifier()
    scores = cross_val_score(dec, x, y, n_jobs=-1, cv=10)
    print(scores.mean())  # 0.9658585858585859

    # 导出决策树并可视化
    dec.fit(x, y)
    tree_dot = export_graphviz(dec)
    graph = pydotplus.graph_from_dot_data(tree_dot)
    graph.write_pdf('export/6-4-FTP_attack_tree.pdf')


if __name__ == "__main__":
    main()
