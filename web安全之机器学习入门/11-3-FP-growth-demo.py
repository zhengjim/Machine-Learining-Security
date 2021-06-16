import pyfpgrowth


def main():
    datasets = [
        [1, 2, 5],
        [2, 4],
        [2, 3],
        [1, 2, 4],
        [1, 3],
        [2, 3],
        [1, 3],
        [1, 2, 3, 5],
        [1, 2, 3]
    ]
    patterns = pyfpgrowth.find_frequent_patterns(datasets, 2)
    rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
    print(rules)


if __name__ == '__main__':
    main()
