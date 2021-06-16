from model import Apriori


def main():
    # 生成测试样本
    data = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    # apriori 算法
    apriori = Apriori()
    frequently_itemset, support_data = apriori.apriori(data, 0.5)
    rules = apriori.generate_rules(frequently_itemset, support_data, min_reliability=0.7)
    print(rules)


if __name__ == '__main__':
    main()
