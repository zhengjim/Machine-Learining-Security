import networkx as nx
import matplotlib.pyplot as plt


# 疑似账号被盗
def hack_account():
    # activesyncid为2的硬件登录了mike和john两个账户。
    # mike历史上曾经成功登录过activesyncid为1的硬件以及activesyncid为2的硬件，初步判定activesyncid为2的硬件盗取了mike的账户登录。
    with open("data/KnowledgeGraph/sample1.txt") as f:
        G = nx.Graph()
        for line in f:
            line = line.strip('\n')
            # 用户名、登录IP地址、手机号、硬件全局唯一activesyncid
            uid, ip, tel, activesyncid = line.split(',')
            G.add_edge(uid, ip)
            G.add_edge(uid, tel)
            G.add_edge(uid, activesyncid)
        nx.draw(G, with_labels=True, node_size=600)
        plt.show()


# 疑似撞库攻击
def attack_pass():
    # 大量账户从ip1登录，并且ua字段相同，登录失败和成功的情况均存在，疑似发生了撞库攻击行为。
    with open("data/KnowledgeGraph/sample2.txt") as f:
        G = nx.Graph()
        for line in f:
            line = line.strip('\n')
            # 用户名、登录IP地址、登录状态、ua头
            uid, ip, login, ua = line.split(',')
            G.add_edge(uid, ip)
            G.add_edge(uid, login)
            G.add_edge(uid, ua)
        nx.draw(G, with_labels=True, node_size=600)
        plt.show()


# 疑似刷单
def click_farming():
    # 虽然两台设备hid1和hid2登录账户不一样， 但是他们共同安装的App2上的登录用户名相同，从而可以判断这两台设备属于同一个人，该人疑似使用这两台设备分别扮演买家和卖家进行刷单行为(这判断方法有点呆)
    G = nx.Graph()
    with open("data/KnowledgeGraph/sample3.txt") as f:
        for line in f:
            line = line.strip('\n')
            # 硬件指纹(唯一标识)、登录用户名、App的名称
            hid, uid, app = line.split(',')
            G.add_edge(hid, uid)
            G.add_edge(hid, app)

    with open("data/KnowledgeGraph/sample4.txt") as f:
        for line in f:
            line = line.strip('\n')
            # 硬件指纹(唯一标识)、登录用户名、用户行为(下单or接单)
            hid, uid, action = line.split(',')
            G.add_edge(hid, uid)
            G.add_edge(hid, action)

    nx.draw(G, with_labels=True, node_size=600)
    plt.show()


def main():
    hack_account()
    attack_pass()
    click_farming()


if __name__ == "__main__":
    main()
