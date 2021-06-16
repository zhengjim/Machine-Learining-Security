import pyfpgrowth


def main():
    datasets = []
    with open("data/KnowledgeGraph/sample7.txt") as f:
        for line in f:
            line = line.strip('\n')
            ip, ua, target = line.split(',')
            datasets.append([ip, ua, target])

    patterns = pyfpgrowth.find_frequent_patterns(datasets, 3)
    rules = pyfpgrowth.generate_association_rules(patterns, 0.9)
    print(rules)


if __name__ == '__main__':
    main()
