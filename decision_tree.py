import numpy as np

class Leaf:
    def __init__(self, data, uncert):
        self.predict = np.sum(data[:,-1]) / float(data.shape[0])
        self.uncert = uncert

class NodeInfo:
    def __init__(self, col, val, uncert):
        self.col = col
        self.val = val
        self.uncert = uncert # gini or entropy
        self.gain = 0

    def checkrow(self, data):
        return data[self.col] >= self.val

    def checkrows(self, data):
        return data[:,self.col] >= self.val

class DecisionNode:
    def __init__(self, left_branch, right_branch, info):
        self.left = left_branch
        self.right = right_branch
        self.info = info

class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=3):
        self.max_depth = max_depth
        if criterion == 'gini':
            self.core = lambda x: 2 * x * (1 - x)
        elif criterion == 'entropy':
            self.core = lambda x: - x * np.log2(x) - (1 - x) * np.log2((1 - x))

    def fit(self, X, y):
        self.branch = self.__build_tree(np.column_stack((X, y)), 0)
        return self.branch

    def score(self, X, y):
        score_error = 0
        for i in range(0, X.shape[0]):
            score_error += abs(y[i] - self.__classify(X[i], self.branch))
        return (X.shape[0] - score_error) / float(X.shape[0])

    def __get_uncertainty(self, data):
        prob = np.sum(data[:,-1]) / data.shape[0]
        return 0 if prob == 0 or prob == 1 else self.core(prob)

    def __get_gain(self, current_uncert, left, right):
        count = left.shape[0] + right.shape[0]
        left_uncert = self.__get_uncertainty(left)
        right_uncert = self.__get_uncertainty(right)
        return current_uncert - left.shape[0] / float(count) * left_uncert - right.shape[0] / float(count) * right_uncert

    def __partition(self, data, info):
        left = data[np.transpose((info.checkrows(data)) == 0)]
        right = data[np.transpose(info.checkrows(data))]
        return left, right

    def __traversal(self, data, uncert):
        optimun_info = None
        for i in range(0, data.shape[1] - 1):
            values = [row[i] for row in data]
            values.sort()

            for val in values:
                info = NodeInfo(i, val, uncert)
                left, right = self.__partition(data, info)
                if len(left) == 0 or len(right) == 0:
                    continue
                info.gain = self.__get_gain(uncert, left, right)
                if optimun_info is None or info.gain > optimun_info.gain:
                    optimun_info = info

        return optimun_info

    def __build_tree(self, data, depth):
        uncert = self.__get_uncertainty(data)

        if depth >= self.max_depth:
            return Leaf(data, uncert)

        info = self.__traversal(data, uncert)
        if info is None:
            return Leaf(data, uncert)

        left, right = self.__partition(data, info)
        return DecisionNode(self.__build_tree(left, depth + 1), self.__build_tree(right, depth + 1), info)

    def __classify(self, data, node):
        if isinstance(node, Leaf):
            return node.predict
        return self.__classify(data, node.right if node.info.checkrow(data) else node.left)
