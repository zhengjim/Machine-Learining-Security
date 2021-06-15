import networkx as nx
import matplotlib.pyplot as plt


# 挖掘后门文件潜在联系
def backfile_cc():
    # 后门文件file1和file2分别对应C&C域名domain1、domain2、domain3、domain6domain2、domain3、 domain4、domain5
    # 其中domain2和domain3同时被file1和file2使用，初步怀疑邮箱file1和file1为同一黑产团体控制的后门文件，
    # domain1至domain4均疑似黑产同时控制，并很可能是同一用途，比如DDoS。
    with open("data/KnowledgeGraph/sample6.txt") as f:
        G = nx.Graph()
        for line in f:
            line = line.strip('\n')
            # 文件(md5)、cc域名
            file, domain = line.split(',')
            G.add_edge(file, domain)

        nx.draw(G, with_labels=True, node_size=600)
        plt.show()


# 挖掘域名潜在联系
def domain_contact():
    # 邮箱mail1和mail2分别注册了域名domain1、domain3和domain2、domain4、domain5，
    # 其中domain1和domain2都指向同一个ip1，domain3和domain4都指向同一个ip2，初步怀疑邮箱mail1和mail2被同一黑产团体控制，
    # domain1至domain4均疑似黑产同时控制， 并很可能是同一用途，比如C&C服务器或者钓鱼网站。
    with open("data/KnowledgeGraph/sample5.txt") as f:
        G = nx.Graph()
        for line in f:
            line = line.strip('\n')
            # 注册邮箱、域名、IP
            email, domain, ip = line.split(',')
            G.add_edge(email, domain)
            G.add_edge(domain, ip)
        nx.draw(G, with_labels=True, node_size=600)
        plt.show()


def main():
    backfile_cc()
    domain_contact()


if __name__ == "__main__":
    main()
