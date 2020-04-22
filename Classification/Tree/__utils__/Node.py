import numpy as np
from scipy.stats import mode
import pandas as pd
# from Algorithms.Tree.DecisionTree import DecisionTree
import Classification.Tree.DecisionTree as DecisionTree
# DecisionTree = Algorithms.Tree.DecisionTree.DecisionTree
# DecisionTree.DecisionTree.

class Node():
    def __init__(self, pts, depth, rulesSoFar=[]):
        self.classProportions = self.calculateClassProportions(pts)
        self.depth = depth
        self.feature = None
        self.feat_i = None
        self.beta = None
        self.left = None
        self.right = None
        self.purity = max(self.classProportions.values(), default=0)
        self.entropy = DecisionTree.DecisionTree.__entropy__(pts)
        self.rulesSoFar = rulesSoFar
        self.mode = -1
        self.size = len(pts)
        self.isQuantitative = -1

    def updateNode(self, pts, feature, feat_i, beta, isQuantitative, left=None, right=None):
        self.feature = feature
        self.feat_i = feat_i
        self.beta = beta
        self.left = left
        self.right = right
        self.isQuantitative = isQuantitative
        if isinstance(beta, tuple):
            self.mode = mode(pts[pd.isnull(pts[:, self.feat_i]) == False, self.feat_i])[0][0]
        else:
            p = list(pts[pd.isnull(pts[:, self.feat_i]) == False, self.feat_i])
            if len(p) > 0:
                self.mode = np.average(p)
            else:
                self.mode = 0

    @property
    def isLeaf(self):
        return self.left is None and self.right is None

    def calculateClassProportions(self, pts):
        d = {}
        for c in set(pts[:, -1]):
            d[c] = np.average(list(pts[:, -1] == c))
        return d

    def traverse(self, pt):
        assert not self.isLeaf, "Traversing down a leaf node is not allowed."
        xi = pt[self.feat_i]
        if pd.isnull(xi):
            xi = self.mode
        if isinstance(self.beta, tuple):
            if xi in self.beta:
                return self.left
            else:
                return self.right
        if xi <= self.beta:
            return self.left
        return self.right

    def classify(self):
        return max(self.classProportions.keys(), key=lambda k: self.classProportions[k], default=0)

    def leftRule(self, cat_reg):
        if self.isQuantitative:
            return f"\"{self.feature}\" ≤ {np.round(self.beta, 2)}"
        return f"\"{self.feature}\" ∈ {set(self.beta)}"

    def rightRule(self, cat_reg):
        if self.isQuantitative:
            return f"\"{self.feature}\" > {np.round(self.beta, 2)}"
        return f"\"{self.feature}\" ∈ {cat_reg[self.feature] - set(self.beta)}"

    def string(self, labelDicts={}):
        c = self.classify()
        cLabel = labelDicts.get(c, c)
        if self.isLeaf:
            return f"{cLabel}\n(p={np.round(self.classProportions.get(c, 0) * 100, 2)}%)\n(n={self.size})"
        if self.isQuantitative:
            return f"Depth: {self.depth}\nSize: {self.size}\nP({cLabel})={np.round(self.classProportions[c] * 100, 2)}%\nSplit Feature: \"{self.feature}\"\nBeta: {np.round(self.beta, 2)}"
        return f"Depth: {self.depth}\nSize: {self.size}\nP({cLabel})={np.round(self.classProportions[c] * 100, 2)}%\nSplit Feature: \"{self.feature}\"\nBeta: {self.beta}"

    @property
    def strID(self):
        return f"{self.depth}-{self.feature}-{self.beta}-{self.size}"

    def pathStr(self, point):
        feat = f"(\"{self.feature}\")"
        xi = point[self.feat_i]
        if pd.isnull(xi):
            xi = self.mode
        if self.isQuantitative:
            if xi < self.beta:
                return f"{feat} ≤ {self.beta} -> Left"
            return f"{feat} > {self.beta} -> Right"
        if xi in self.beta:
            return f"{feat} is in {set(self.beta)} -> Left"
        return f"{feat} is not in {set(self.beta)} -> Right"