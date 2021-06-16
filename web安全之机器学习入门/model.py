class Apriori:
    """  Apriori算法  """

    def apriori(self, data, min_support=0.5):
        """apriori算法实现

        :param data: 数据集
        :param min_support: 最小支持度
        :return: 频繁项集，频繁项集的支持度
        """
        # 获取非频繁项集
        itemset_1 = self.create_itemset(data)
        # 转化事务集的形式，每个元素都转化为集合。
        data = list(map(set, data))
        # 获取频繁1项集和对应的支持度
        frequently_itemset_1, support_data = self.scan_data(data, itemset_1, min_support)
        # frequently_itemset用来存储所有的频繁项集
        frequently_itemset = [frequently_itemset_1]

        k = 2
        # 一直迭代到项集数目过大而在事务集中不存在这种n项集
        while len(frequently_itemset[k - 2]) > 0:
            # 根据频繁项集生成新的候选项集
            itemset_k = self.create_new_itemset(frequently_itemset[k - 2], k)
            frequently_itemset_k, support_k = self.scan_data(data, itemset_k, min_support)
            support_data.update(support_k)
            frequently_itemset.append(frequently_itemset_k)
            k += 1
        return frequently_itemset, support_data

    def create_itemset(self, data):
        """创建元素为1的项集

        :param data: 原始数据集
        :return:创建元素为1的项集
        """
        itemset = []
        # 元素个数为1的项集（非频繁项集，因为还没有同最小支持度比较）
        for items in data:
            for item in items:
                if [item] not in itemset:
                    itemset.append([item])
        itemset.sort()  # 这里排序是为了，生成新的候选集时可以直接认为两个n项候选集前面的部分相同
        # 因为除了候选1项集外其他的候选n项集都是以二维列表的形式存在，所以要将候选1项集的每一个元素都转化为一个单独的集合
        return list(map(frozenset, itemset))  # list(map(frozenset, itemset))的语义是将C1由列表转换为不变集合

    def scan_data(self, data, k, min_support):
        """找出候选集中的频繁项集

        :param data: 全部数据集
        :param k: 为大小为包含k个元素的候选项集
        :param min_support: 设定的最小支持度
        :return: frequently_itemset为在k中找出的频繁项集（支持度大于min_support的），support_data记录各频繁项集的支持度
        """
        scan_itemset = {}
        for i in data:
            for j in k:
                if j.issubset(i):
                    scan_itemset[j] = scan_itemset.get(j, 0) + 1  # 计算每一个项集出现的频率
        items_num = float(len(list(data)))
        frequently_itemset = []
        support_data = {}
        for key in scan_itemset:
            support = scan_itemset[key] / items_num
            if support >= min_support:
                frequently_itemset.insert(0, key)  # 将频繁项集插入返回列表的首部
            support_data[key] = support
        return frequently_itemset, support_data

    def create_new_itemset(self, frequently_itemset, k):
        """通过频繁项集列表frequently_itemset和项集个数k生成候选项集

        :param frequently_itemset: 频繁项集列表
        :param k: 项集个数
        :return: 候选项集
        """
        new_frequently_itemset = []
        frequently_itemset_len = len(frequently_itemset)
        for i in range(frequently_itemset_len):
            for j in range(i + 1, frequently_itemset_len):
                # 前k-1项相同时，才将两个集合合并，合并后才能生成k+1项
                l1 = list(frequently_itemset[i])[: k - 2]
                l2 = list(frequently_itemset[j])[: k - 2]  # 取出两个集合的前k-1个元素
                l1.sort()
                l2.sort()
                if l1 == l2:
                    new_frequently_itemset.append(frequently_itemset[i] | frequently_itemset[j])
        return new_frequently_itemset

    def calc_reliability(self, freq_set, h, support_data, brl, min_reliability=0.7):
        """对候选规则集进行评估

        :param freq_set: 频繁项集
        :param h: 元素列表
        :param support_data: 项集的支持度
        :param brl: 生成的关联规则
        :param min_reliability: 最小置信度
        :return: 规则列表的右部, candidate_rule_set()中用到
        """
        pruned = []
        for conseq in h:
            conf = support_data[freq_set] / support_data[freq_set - conseq]
            if conf >= min_reliability:
                brl.append((freq_set - conseq, conseq, conf))
                pruned.append(conseq)
        return pruned

    def candidate_rule_set(self, freq_set, h, support_data, brl, min_reliability=0.7):
        """生成候选规则集

        :param freq_set: 频繁项集
        :param h: 元素列表
        :param support_data: 项集的支持度
        :param brl: 生成的关联规则
        :param min_reliability: 最小置信度
        :return: 下一层候选规则集
        """
        m = len(h[0])
        if len(freq_set) > m + 1:
            hmp1 = self.create_new_itemset(h, m + 1)
            hmp1 = self.calc_reliability(freq_set, hmp1, support_data, brl, min_reliability)
            if len(hmp1) > 1:
                self.candidate_rule_set(freq_set, hmp1, support_data, brl, min_reliability)

    def generate_rules(self, frequently_itemset, support_data, min_reliability=0.7):
        """关联规则生成

        :param frequently_itemset: 频繁项集
        :param support_data: 频繁项集的支持度
        :param min_reliability: 最小置信度
        :return: 包含可信度的规则列表
        """
        big_rule_list = []
        for i in range(1, len(frequently_itemset)):
            for freq_set in frequently_itemset[i]:
                h1 = [frozenset([item]) for item in freq_set]
                if i > 1:
                    # 三个及以上元素的集合
                    self.candidate_rule_set(freq_set, h1, support_data, big_rule_list, min_reliability)
                else:
                    # 两个元素的集合
                    self.calc_reliability(freq_set, h1, support_data, big_rule_list, min_reliability)
        return big_rule_list
