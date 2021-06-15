from neo4j import GraphDatabase, basic_auth


# neo4j直接使用了沙盒 https://sandbox.neo4j.com/
# pip install neo4j-driver
# pip install JPype1


def main():
    # 自行修改
    driver = GraphDatabase.driver(
        "bolt://x.x.x.x:7687",
        auth=basic_auth("neo4j", "transmitters-amusements-saturdays"))
    session = driver.session()

    # 插入数据
    # Insert data
    insert_query = '''
    UNWIND $pairs as pair
    MERGE (p1:Person {name:pair[0]})
    MERGE (p2:Person {name:pair[1]})
    MERGE (p1)-[:KNOWS]-(p2);
    '''
    data = [["Jim", "Mike"], ["Jim", "Billy"], ["Anna", "Jim"], ["Anna", "Mike"], ["Sally", "Anna"], ["Joe", "Sally"],
            ["Joe", "Bob"], ["Bob", "Sally"]]
    session.run(insert_query, parameters={"pairs": data})

    # 朋友的朋友
    foaf_query = '''
    MATCH (person:Person)-[:KNOWS]-(friend)-[:KNOWS]-(foaf)
    WHERE person.name = $name
      AND NOT (person)-[:KNOWS]-(foaf)
    RETURN foaf.name AS name
    '''
    results = session.run(foaf_query, parameters={"name": "Joe"})
    for record in results:
        print(record["name"])

    # 共同的朋友
    common_friends_query = """
    MATCH (user:Person)-[:KNOWS]-(friend)-[:KNOWS]-(foaf:Person)
    WHERE user.name = $user AND foaf.name = $foaf
    RETURN friend.name AS friend
    """
    results = session.run(common_friends_query, parameters={"user": "Joe", "foaf": "Sally"})
    for record in results:
        print(record["friend"])

    # 连接路径
    connecting_paths_query = """
    MATCH path = shortestPath((p1:Person)-[:KNOWS*..6]-(p2:Person))
    WHERE p1.name = $name1 AND p2.name = $name2
    RETURN path
    """

    results = session.run(connecting_paths_query, parameters={"name1": "Joe", "name2": "Billy"})
    for record in results:
        print(record["path"])

    session.close()


if __name__ == "__main__":
    main()
