# 预处理与特征提取
# 数字型直接可以用，但要进行预处理

import numpy as np

X = np.array([
    [1., -1., 2.],
    [2., 0., 0.],
    [0., 1., -1.]]
)



from sklearn import preprocessing

# 标准化预处理

# x_scaled = preprocessing.scale(X)
# print(x_scaled)

scaled = preprocessing.StandardScaler()
x_scaled = scaled.fit_transform(X)
print(x_scaled)

# 字符串，特征提取 本质上是做单词切分，不同的单词当作一个新的特征

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# measurements = [
#     {'city': 'Dubai', 'temperature': 33.},
#     {'city': 'London', 'temperature': 12.},
#     {'city': 'San Fransisco', 'temperature': 18.},
# ]
#
# dict = DictVectorizer()
# m = dict.fit_transform(measurements).toarray()
# print(dict.get_feature_names())

cv = CountVectorizer()
data = cv.fit_transform(["life is short,i like python python", "life is too long,i dislike python"])
print(cv.get_feature_names())
print(data.toarray())

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus

iris = load_iris()
x = iris.data
y = iris.target

dec = DecisionTreeClassifier()
dec.fit(x, y)

tree_dot = export_graphviz(dec, out_file=None)

print(dec.predict([[1, 2]]))
