import numpy as np
import pandas as pd
from scipy.stats import mode
import Classification.Tree.__utils__.Node as Node
from Models import ClassificationModel

Node = Node.Node

class DecisionTree(ClassificationModel):
    def __init__(self, column_names=[], categorical_cols=[], cols_to_ignore=[], verbose=False, normalize_columns=False, normalize_rows=False, addBias=False):
        super().__init__(verbose=verbose, normalize_columns=normalize_columns, normalize_rows=normalize_rows, addBias=addBias)
        # assert len(column_names) == training_data.shape[1], "Column Names must be same length as data dimension including label!"
        self.column_names = column_names
        self.categorical_cols = categorical_cols
        self.cols_to_ignore = cols_to_ignore
        self.quantitative_cols = [c for c in column_names
                                  if c not in categorical_cols
                                  and c != column_names[-1]
                                  and c not in cols_to_ignore]
        self.columnsSpecified = len(column_names) != 0
        self.trained = False
        self.name = "DecisionTree"

    def train(self, data, maxDepth=float("inf"), minPurity=1, nodeSize=5, num_features=-1):
        # assert not self.trained, "Decision Tree can only be trained once. Please re-initialize to train again."
        self.__registerColumns__(data)
        if num_features == -1: num_features = len(self.column_names) - 1
        self.maxDepth = maxDepth
        self.purity = minPurity
        self.minNodeSize = nodeSize
        self.root = self.__growTree__(data, 0, num_features)
        self.trained = True
        return self

    def __registerColumns__(self, data):
        if len(self.column_names) == 0:
            # If no column names passed in, assume that all columns are quantitative
            self.column_names = [str(i) for i in range(data.shape[1])]
            self.quantitative_cols = self.column_names

        self.categorical_register = {}
        for c in self.categorical_cols:
            c_i = self.column_names.index(c)
            self.categorical_register[c] = set(data[:, c_i])

    def __growTree__(self, pts, depth, num_features, rulesSoFar=[]):
        node = Node(pts, depth, rulesSoFar)
        if self.__shouldStop__(depth, node):
            return node

        feature, beta = self.__bestSplit__(pts, num_features)
        if feature == -1:  # or (feature, beta) in rulesSoFar:
            return node

        feat_i = self.column_names.index(feature)

        if feature in self.categorical_cols:
            leftPtsIdx = np.isin(pts[:, feat_i], beta)
        else:
            leftPtsIdx = pts[:, feat_i] <= beta
        rightPtsIdx = leftPtsIdx == False

        if DecisionTree.__averageEntropy__(pts, leftPtsIdx, rightPtsIdx) == node.entropy:
            return node

        left = self.__growTree__(pts[leftPtsIdx], depth + 1, num_features, rulesSoFar + [(feature, beta)])
        right = self.__growTree__(pts[rightPtsIdx], depth + 1, num_features, rulesSoFar + [(feature, beta)])

        node.updateNode(pts, feature, feat_i, beta, feature in self.quantitative_cols, left, right)
        return node

    def __shouldStop__(self, depth, node):
        return depth >= self.maxDepth or node.purity >= self.purity or node.size <= self.minNodeSize

    def __bestSplit__(self, pts, num_features):
        """Returns best_feature, best_beta.
        For quantitative features, beta is a cutoff. If feature is less than beta, assign to left. Otherwise, right.
        For qualitative, beta is set of labels on left node. If feature is in beta, assign left. Otherwise, right."""
        lowestEntropy = float("inf")
        bestBeta = -1
        bestFeature = -1
        featsToTest = np.random.choice(self.column_names[:-1], num_features, False)

        for c in featsToTest:
            if c in self.quantitative_cols:
                ent, beta = self.__bestBetaForQuantFeature__(pts, c)
                if ent < lowestEntropy:
                    lowestEntropy = ent
                    bestBeta = beta
                    bestFeature = c
            elif c in self.categorical_cols:
                ent, beta = self.__bestBetaForCatFeature__(pts, c)
                if ent < lowestEntropy:
                    lowestEntropy = ent
                    bestBeta = beta
                    bestFeature = c
        return bestFeature, bestBeta

    def __bestBetaForCatFeature__(self, pts, feat):
        feat_i = self.column_names.index(feat)
        nan_idx = self.__impute__(pts, feat_i, False)
        values = pts[:, feat_i]

        lowestEntropy = float("inf")
        bestBeta = ()
        for leftGroup in self.__allGroupings__(set(values)):
            leftPtsIdx = np.isin(values, leftGroup)
            rightPtsIdx = leftPtsIdx == False
            weightedEntropy = DecisionTree.__averageEntropy__(pts, leftPtsIdx, rightPtsIdx)
            if weightedEntropy < lowestEntropy:
                lowestEntropy = weightedEntropy
                bestBeta = leftGroup

        pts[nan_idx, feat_i] = float("nan")
        return lowestEntropy, bestBeta

    def __bestBetaForQuantFeature__(self, pts, feat):
        feat_i = self.column_names.index(feat)
        nan_idx = self.__impute__(pts, feat_i, True)
        feat_values = sorted(set(pts[:, feat_i]))
        sortedByFeat = pts[np.argsort(pts[:, feat_i])]
        total = len(pts)
        lowestEntropy = float("inf")
        bestBeta = -1

        feat_vals = np.unique(sortedByFeat[:, feat_i], return_counts=True)
        feat_val_counts = np.split(sortedByFeat[:, -1], np.cumsum(feat_vals[1]))[:-1]
        l_0 = 0
        l_1 = 0
        r_1 = sum(sortedByFeat[:, -1])
        r_0 = len(sortedByFeat) - r_1
        for beta_i, c in zip(feat_vals[0], feat_val_counts):
            new_1 = sum(c)
            new_0 = len(c) - sum(c)
            l_1 += new_1
            r_1 -= new_1
            l_0 += new_0
            r_0 -= new_0

            total_l = l_0 + l_1
            if total_l == 0: total_l = 1

            total_r = r_0 + r_1
            if total_r == 0: total_r = 1

            p0_l = l_0 / total_l
            p1_l = 1 - p0_l
            p0_r = r_0 / total_r
            p1_r = 1 - p0_r

            ent_left = -(p0_l * np.log2(np.clip(p0_l, 1e-15, 1))) + -(p1_l * np.log2(np.clip(p1_l, 1e-15, 1)))
            ent_right = -(p0_r * np.log2(np.clip(p0_r, 1e-15, 1))) + -(p1_r * np.log2(np.clip(p1_r, 1e-15, 1)))
            weighted_ent = ((l_0 + l_1) * ent_left + (r_0 + r_1) * ent_right) / (total)

            if weighted_ent < lowestEntropy:
                lowestEntropy = weighted_ent
                bestBeta = beta_i
        pts[nan_idx, feat_i] = float("nan")
        return lowestEntropy, bestBeta

    def __impute__(self, pts, feat_i, isQuantitative):
        try:
            nans = pd.isnull(pts[:, feat_i])
            if isQuantitative:
                goodpts = pts[nans == False]
                mean_val = 0
                if len(goodpts) > 0:
                    mean_val = np.mean(list(pts[nans == False, feat_i]))
                pts[nans, feat_i] = mean_val
            else:
                mode_val = mode(pts[nans == False, feat_i])[0][0]
                pts[nans, feat_i] = mode_val
            return nans
        except:
            return []

    def __allGroupings__(self, values):
        """returns a list of tuples, where each tuple is the items in the left subtree"""
        return [tuple([x]) for x in values if not pd.isnull(x)]

    @staticmethod
    def __averageEntropy__(pts, leftIdx, rightIdx):
        weightedEntropy = 0
        weightedEntropy += sum(leftIdx) * DecisionTree.__entropy__(pts[leftIdx, :])
        weightedEntropy += sum(rightIdx) * DecisionTree.__entropy__(pts[rightIdx, :])
        weightedEntropy /= len(pts)
        return weightedEntropy

    @staticmethod
    def __entropy__(pts):
        classes = pts[:, -1]
        s = 0
        for c in set(classes):
            pc = np.average(classes == c)
            s -= (pc * np.log2(pc))
        return s

    def predict(self, data : np.ndarray):
        assert data.shape[1] in [len(self.column_names), len(self.column_names) - 1], "Wrong input size!"
        return np.array([self.classifySinglePoint(p) for p in data])

    def __predictSinglePoint__(self, point):
        assert self.trained, "Must be trained before classifying point!"
        currNode = self.root
        while not currNode.isLeaf:
            currNode = currNode.traverse(point)
        return currNode.classify()

    #
    def classify(self, X):
        assert X.shape[1] in [len(self.column_names), len(self.column_names) - 1], "Wrong input size!"
        return np.array([self.classifySinglePoint(p) for p in X])


# class DecisionTreeOLDDDDDDD():
#     def __init__(self, training_data, column_names, categorical_cols=[], cols_to_ignore=[]):
#         assert len(column_names) == training_data.shape[
#             1], "Column Names must be same length as data dimension including label!"
#         self.training_data = training_data.copy()
#         self.training_data = training_data[pd.isnull(training_data[:, -1]) == False]
#         self.column_names = column_names
#         self.categorical_cols = categorical_cols
#         self.quantitative_cols = [c for c in column_names
#                                   if c not in categorical_cols
#                                   and c != column_names[-1]
#                                   and c not in cols_to_ignore]
#         self.cols_to_ignore = cols_to_ignore
#         self.trained = False
#         self.registerCategoricalValues()
#
#     def registerCategoricalValues(self):
#         self.categorical_register = {}
#         for c in self.categorical_cols:
#             c_i = self.column_names.index(c)
#             self.categorical_register[c] = set(self.training_data[:, c_i])
#
#     def train(self, maxDepth=float("inf"), purity=1, minNodeSize=5, num_features=-1):
#         assert not self.trained, "Decision Tree can only be trained once. Please re-initialize to train again."
#         if num_features == -1: num_features = len(self.column_names) - 1
#         self.maxDepth = maxDepth
#         self.purity = purity
#         self.minNodeSize = minNodeSize
#         self.root = self.growTree(self.training_data, 0, num_features)
#         self.trained = True
#         self.training_data = None
#         return self
#
#     def shouldStop(self, pts, depth, node):
#         return depth >= self.maxDepth or node.purity >= self.purity or node.size <= self.minNodeSize
#
#     @staticmethod
#     def entropy(pts):
#         classes = pts[:, -1]
#         s = 0
#         for c in set(classes):
#             pc = np.average(classes == c)
#             s -= (pc * np.log2(pc))
#         return s
#
#     def impute(self, pts, feat_i, isQuantitative):
#         try:
#             nans = pd.isnull(pts[:, feat_i])
#             if isQuantitative:
#                 goodpts = pts[nans == False]
#                 mean_val = 0
#                 if len(goodpts) > 0:
#                     mean_val = np.mean(list(pts[nans == False, feat_i]))
#                 pts[nans, feat_i] = mean_val
#             else:
#                 mode_val = mode(pts[nans == False, feat_i])[0][0]
#                 pts[nans, feat_i] = mode_val
#             return nans
#         except:
#             return []
#
#     def allGroupings(self, values):
#         """returns a list of tuples, where each tuple is the items in the left subtree"""
#         return [tuple([x]) for x in values if not pd.isnull(x)]  # groupings
#
#     def bestBetaForCatFeature(self, pts, feat):
#         feat_i = self.column_names.index(feat)
#         nan_idx = self.impute(pts, feat_i, False)
#         values = pts[:, feat_i]
#
#         lowestEntropy = float("inf")
#         bestBeta = ()
#         for leftGroup in self.allGroupings(set(values)):
#             leftPtsIdx = np.isin(values, leftGroup)
#             rightPtsIdx = leftPtsIdx == False
#             weightedEntropy = DecisionTree.averageEntropy(pts, leftPtsIdx, rightPtsIdx)
#             if weightedEntropy < lowestEntropy:
#                 lowestEntropy = weightedEntropy
#                 bestBeta = leftGroup
#
#         pts[nan_idx, feat_i] = float("nan")
#         return lowestEntropy, bestBeta
#
    def classifySinglePoint(self, point):
        assert self.trained, "Must be trained before classifying point!"
        currNode = self.root
        while not currNode.isLeaf:
            currNode = currNode.traverse(point)
        return currNode.classify()
#
    def classify(self, X):
        assert X.shape[1] in [len(self.column_names), len(self.column_names) - 1], "Wrong input size!"
        return np.array([self.classifySinglePoint(p) for p in X])
#
#     def accuracy(self, X):
#         assert X.shape[1] == len(self.column_names), "Wrong input size!"
#         return np.average(self.classify(X) == X[:, -1])
#
#     def pathThroughTree(self, point, output_map={}):
#         assert self.trained, "Must be trained before classifying point!"
#         s = ""
#         currNode = self.root
#         while not currNode.isLeaf:
#             s += f"Depth {currNode.depth}: {currNode.pathStr(point)}\n"
#             currNode = currNode.traverse(point)
#         s += f"Therefor, we classify this as {output_map.get(currNode.classify(), currNode.classify())}"
#         return s
#
#     def visualize(self, labelDict={}):
#         return self.visualizeHelper(self.root, labelDict=labelDict)
#
#     def visualizeHelper(self, node, dot=None, parent=None, left=False, labelDict={}):
#         if node == True: node = self.root
#         if dot is None: dot = Digraph(f"{node.strID} tree visualization")
#         if node is None:
#             return
#         dot.node(node.strID, node.string(labelDict), fontsize="8")
#         if parent:
#             if left:
#                 dot.edge(parent.strID, node.strID, label=f"{parent.leftRule(self.categorical_register)}", fontsize="8")
#             else:
#                 dot.edge(parent.strID, node.strID, label=f"{parent.rightRule(self.categorical_register)}", fontsize="8")
#
#         self.visualizeHelper(node.left, dot, node, True, labelDict)
#         self.visualizeHelper(node.right, dot, node, False, labelDict)
#         return Source(dot)
#
#     def growTree(self, pts, depth, num_features, rulesSoFar=[]):
#         node = Node(pts, depth, rulesSoFar)
#         if self.shouldStop(pts, depth, node):
#             return node
#
#         feature, beta = self.bestSplit(pts, num_features)
#         if feature == -1:  # or (feature, beta) in rulesSoFar:
#             return node
#
#         feat_i = self.column_names.index(feature)
#         leftPtsIdx = None
#         if feature in self.categorical_cols:
#             leftPtsIdx = np.isin(pts[:, feat_i], beta)
#         else:
#             leftPtsIdx = pts[:, feat_i] <= beta
#         rightPtsIdx = leftPtsIdx == False
#
#         if DecisionTree.averageEntropy(pts, leftPtsIdx, rightPtsIdx) == node.entropy:
#             return node
#
#         left = self.growTree(pts[leftPtsIdx], depth + 1, num_features, rulesSoFar + [(feature, beta)])
#         right = self.growTree(pts[rightPtsIdx], depth + 1, num_features, rulesSoFar + [(feature, beta)])
#
#         node.updateNode(pts, feature, feat_i, beta, feature in self.quantitative_cols, left, right)
#         return node
#
#     def bestSplit(self, pts, num_features):
#         """Returns best_feature, best_beta.
#         For quantitative features, beta is a cutoff. If feature is less than beta, assign to left. Otherwise, right.
#         For qualitative, beta is set of labels on left node. If feature is in beta, assign left. Otherwise, right."""
#         lowestEntropy = float("inf")
#         bestBeta = -1
#         bestFeature = -1
#         featsToTest = np.random.choice(self.column_names[:-1], num_features, False)
#
#         for c in featsToTest:  # TODO incvlude ALL cols not just quantitatie
#             if c in self.quantitative_cols:
#                 ent, beta = self.bestBetaForQuantFeature(pts, c)
#                 if ent < lowestEntropy:
#                     lowestEntropy = ent
#                     bestBeta = beta
#                     bestFeature = c
#             elif c in self.categorical_cols:
#                 ent, beta = self.bestBetaForCatFeature(pts, c)
#                 if ent < lowestEntropy:
#                     lowestEntropy = ent
#                     bestBeta = beta
#                     bestFeature = c
#         return bestFeature, bestBeta
#
#     def bestBetaForQuantFeature(self, pts, feat):
#         feat_i = self.column_names.index(feat)
#         nan_idx = self.impute(pts, feat_i, True)
#         feat_values = sorted(set(pts[:, feat_i]))
#         sortedByFeat = pts[np.argsort(pts[:, feat_i])]
#         total = len(pts)
#         lowestEntropy = float("inf")
#         bestBeta = -1
#
#         feat_vals = np.unique(sortedByFeat[:, feat_i], return_counts=True)
#         feat_val_counts = np.split(sortedByFeat[:, -1], np.cumsum(feat_vals[1]))[:-1]
#         l_0 = 0
#         l_1 = 0
#         r_1 = sum(sortedByFeat[:, -1])
#         r_0 = len(sortedByFeat) - r_1
#         for beta_i, c in zip(feat_vals[0], feat_val_counts):
#             new_1 = sum(c)
#             new_0 = len(c) - sum(c)
#             l_1 += new_1
#             r_1 -= new_1
#             l_0 += new_0
#             r_0 -= new_0
#
#             total_l = l_0 + l_1
#             if total_l == 0: total_l = 1
#
#             total_r = r_0 + r_1
#             if total_r == 0: total_r = 1
#
#             p0_l = l_0 / total_l
#             p1_l = 1 - p0_l
#             p0_r = r_0 / total_r
#             p1_r = 1 - p0_r
#
#             ent_left = -(p0_l * np.log2(np.clip(p0_l, 1e-15, 1))) + -(p1_l * np.log2(np.clip(p1_l, 1e-15, 1)))
#             ent_right = -(p0_r * np.log2(np.clip(p0_r, 1e-15, 1))) + -(p1_r * np.log2(np.clip(p1_r, 1e-15, 1)))
#             weighted_ent = ((l_0 + l_1) * ent_left + (r_0 + r_1) * ent_right) / (total)
#
#             if weighted_ent < lowestEntropy:
#                 lowestEntropy = weighted_ent
#                 bestBeta = beta_i
#         pts[nan_idx, feat_i] = float("nan")
#         return lowestEntropy, bestBeta
#
#     @staticmethod
#     def averageEntropy(pts, leftIdx, rightIdx):
#         weightedEntropy = 0
#         weightedEntropy += sum(leftIdx) * DecisionTree.entropy(pts[leftIdx, :])
#         weightedEntropy += sum(rightIdx) * DecisionTree.entropy(pts[rightIdx, :])
#         weightedEntropy /= len(pts)
#         return weightedEntropy
