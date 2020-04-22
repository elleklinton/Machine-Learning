import numpy as np
import Classification.Tree.__utils__.Node as Node
import Classification.Tree.DecisionTree as DecisionTree
from Models import ClassificationModel

Node = Node.Node
DecisionTree = DecisionTree.DecisionTree

class RandomForest(ClassificationModel):
    def __init__(self, column_names=[], categorical_cols=[], cols_to_ignore=[], verbose=True, normalize_columns=True, normalize_rows=True, addBias=False):
        super().__init__(verbose=verbose, normalize_columns=normalize_columns, normalize_rows=normalize_rows, addBias=addBias)
        self.column_names = column_names
        self.categorical_cols = categorical_cols
        self.cols_to_ignore = cols_to_ignore
        self.quantitative_cols = [c for c in column_names
                                  if c not in categorical_cols
                                  and c != column_names[-1]
                                  and c not in cols_to_ignore]
        self.columnsSpecified = len(column_names) != 0
        self.trained = False
        self.name = "RandomForest"

    def __registerColumns__(self, data):
        if len(self.column_names) == 0:
            # If no column names passed in, assume that all columns are quantitative
            self.column_names = [str(i) for i in range(data.shape[1])]
            self.quantitative_cols = self.column_names

        self.categorical_register = {}
        for c in self.categorical_cols:
            c_i = self.column_names.index(c)
            self.categorical_register[c] = set(data[:, c_i])

    def train(self, data, maxDepth=float("inf"), minPurity=1, nodeSize=5, num_features=-1, num_trees=5):
        assert num_trees > 0, "Number of trees must be greater than 0!"
        assert not self.trained, "Random Forest cannot be already trained!"
        self.__registerColumns__(data)
        if num_features == -1: num_features = int(np.sqrt(len(self.column_names) - 1)) + 1
        self.trees = []
        for i in range(num_trees):
            bootstrap_sample = data[
                np.random.choice(len(data),
                                 len(data), replace=True)
            ]
            t = DecisionTree(self.column_names, self.categorical_cols, self.cols_to_ignore, False, self.normalize_columns, self.normalize_rows, self.addBias)
            # t = DecisionTree(bootstrap_sample, self.column_names, self.categorical_cols, self.cols_to_ignore)
            t.train(data, maxDepth, minPurity, nodeSize, num_features)
            # t.train(maxDepth, purity, minNodeSize, num_features)
            self.trees.append(t)
            if self.verbose: print(f"Trained tree #{i + 1}")

        self.trained = True
        if self.verbose: print(f"Finished Training {num_trees} Trees, m={num_features} features per node!")
        if self.verbose: print(f"Training Accuracy: {np.round(self.accuracy(data) * 100, 2)}%")

    def predict(self, data : np.ndarray):
        assert data.shape[1] in [len(self.column_names), len(self.column_names) - 1], "Wrong input size!"
        assert self.trained, "Random Forest is not trained!"
        return np.array([self.__predictSinglePoint__(p) for p in data])

    def __predictSinglePoint__(self, point):
        assert len(self.trees) != 0, "Model must be trained on at least 1 tree!"
        votes = {}
        for t in self.trees:
            pred = t.classifySinglePoint(point)
            votes[pred] = votes.get(pred, 0) + 1
        return max(votes.keys(), key=lambda k: votes[k], default=0)