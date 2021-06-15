from model import Apriori
import re


def main():
    datasets = []

    with open("data/XSS/xss-2000.txt") as f:
        for line in f:
            index = line.find("?")
            if index > 0:
                line = line[index + 1:len(line)]
                tokens = re.split('\=|&|\?|\%3e|\%3c|\%3E|\%3C|\%20|\%22|<|>|\\n|\(|\)|\'|\"|;|:|,|\%28|\%29', line)
                datasets.append(tokens)

    # apriori 算法
    apriori = Apriori()
    frequently_itemset, support_data = apriori.apriori(datasets, 0.15)
    rules = apriori.generate_rules(frequently_itemset, support_data, min_reliability=1)
    print(rules)


if __name__ == '__main__':
    main()
