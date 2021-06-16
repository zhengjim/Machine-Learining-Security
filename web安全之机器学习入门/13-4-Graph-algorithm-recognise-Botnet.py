from datasets import Datasets
import networkx as nx
import matplotlib.pyplot as plt

# 相似度
N = 0.5
# 黑客团伙IP最少个数
M = 3
# 黑客IP攻击目标最小个数
R = 2


# jarccard系数(交集与并集的个数)
def get_len(d1, d2):
    ds1 = set()
    for d in d1.keys():
        ds1.add(d)

    ds2 = set()
    for d in d2.keys():
        ds2.add(d)
    return len(ds1 & ds2) / len(ds1 | ds2)


def main():
    ip_list = Datasets.load_secrepo()
    good_ip_list = {}
    G = nx.Graph()

    # 攻击的域名超过R的IP才列入统计范围
    for ip in ip_list.keys():
        if len(ip_list[ip]) >= R:
            good_ip_list[ip] = 1

    # 满足阈值的IP导入图数据库
    for ip1 in ip_list.keys():
        for ip2 in ip_list.keys():
            if not ip1 == ip2:
                weight = get_len(ip_list[ip1], ip_list[ip2])
                if (weight >= N) and (ip1 in good_ip_list.keys()) and (ip2 in good_ip_list.keys()):
                    # 点不存在会自动添加
                    G.add_edge(ip1, ip2, weight=weight)

    # 连通分量数目
    n_sub_graphs = nx.number_connected_components(G)
    # 最大连通子图
    sub_graphs = list(G.subgraph(c) for c in nx.connected_components(G))

    # 当同一团伙的IP大于等于M时才显示结果
    for i, sub_graph in enumerate(sub_graphs):
        n_nodes = len(sub_graph.nodes())
        if n_nodes >= M:
            print("Subgraph {0} has {1} nodes {2}".format(i, n_nodes, sub_graph.nodes()))

    nx.draw(G)
    plt.show()


if __name__ == "__main__":
    main()
