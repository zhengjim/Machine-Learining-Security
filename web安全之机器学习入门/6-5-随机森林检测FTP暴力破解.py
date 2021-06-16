from sklearn.ensemble import RandomForestClassifier
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

    # 随机森林 交叉验证
    """
    n_estimators: 多少颗树
    max_depth: 树的最大深度 如果值为None，那么会扩展节点，直到所有的叶子是纯净的，或者直到所有叶子包含少于min_sample_split的样本
    min_samples_split：分割内部节点所需要的最小样本数量
    random_state：random_state是随机数生成器使用的种子
    """
    dec = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(dec, x, y, n_jobs=-1, cv=10)
    print(scores.mean())  # 0.9829090909090908


if __name__ == "__main__":
    main()
