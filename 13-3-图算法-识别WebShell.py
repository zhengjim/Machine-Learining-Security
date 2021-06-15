from neo4j import GraphDatabase, basic_auth
import re

# 先清空数据库，确保在一个空白的环境 match (n) detach delete n

# 导入数据
def import_data(session):
    nodes = {}
    index = 1

    # 读取数据
    file_object = open('data/r-graph.txt', 'r')
    try:
        for line in file_object:
            match_obj = re.match(r'(\S+) -> (\S+)', line, re.M | re.I)
            if match_obj:
                path = match_obj.group(1)
                ref = match_obj.group(2)

                # 节点不存在则新建
                if path not in nodes.keys():
                    path_node = "Page%d" % index
                    nodes[path] = path_node
                    sql = "create (%s:Page {url:\"%s\" , id:\"%d\",in:0,out:0})" % (path_node, path, index)
                    index = index + 1
                    session.run(sql)
                    # print(sql)

                # 节点不存在新建
                if ref not in nodes.keys():
                    ref_node = "Page%d" % index
                    nodes[ref] = ref_node
                    sql = "create (%s:Page {url:\"%s\",id:\"%d\",in:0,out:0})" % (ref_node, ref, index)
                    index = index + 1
                    session.run(sql)
                    # print(sql)

                # 关联关系
                sql = "MATCH (out:Page {url:\"%s\"}), (in:Page {url:\"%s\"}) MERGE (out)-[:IN]->(in)" % (path, ref)
                # sql = "match (%s)-[:IN]->(%s)" % (path_node, ref_node)
                session.run(sql)
                # print(sql)

                # 出度
                sql = "match (n:Page {url:\"%s\"}) SET n.out=n.out+1" % path
                session.run(sql)
                # print(sql)

                # 入度
                sql = "match (n:Page {url:\"%s\"}) SET n.in=n.in+1" % ref
                session.run(sql)
                # print(sql)
    finally:
        file_object.close()


def main():
    # 连接数据库
    driver = GraphDatabase.driver(
        "bolt://52.90.194.108:7687",
        auth=basic_auth("neo4j", "transmitters-amusements-saturdays"))
    session = driver.session()

    # 导入数据
    import_data(session)

    # 查询入度为1出度均为0的节点或者查询入度出度均为1且指向自己的节点
    sql = "MATCH (n:Page) where (n.in=1 and n.out=0) or (n.in=1 and n.out=1) RETURN n.url"
    results = session.run(sql)
    for result in results:
        print("疑是webshell: %s" % result["n.url"])

    # 关闭连接
    session.close()


if __name__ == "__main__":
    main()
