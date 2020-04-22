import numpy as np
np.random.seed(420)

from Tests.Datasets import BreastCancer
from Models import CrossValidationParameter
from Classification.Gaussian.LDA import LDA
from Classification.Gaussian.QDA import QDA
from Classification.LogisticRegression import LogisticRegression
from Classification.Tree.DecisionTree import DecisionTree
from Classification.Tree.RandomForest import RandomForest

dataset = BreastCancer()
train = dataset.train
val = dataset.val


### LDA TESTS ###
model = LDA(verbose=False)
model.train(train)
print(f"{dataset.name} {dataset.datatype} Training Accuracy ({model.name}): {model.accuracy(train)}")
print(f"{dataset.name} {dataset.datatype} Validation Accuracy ({model.name}): {model.accuracy(val)}")


### QDA Tests ###
model = QDA(verbose=False)
model.train(train)
print(f"{dataset.name} {dataset.datatype} Training Accuracy ({model.name}): {model.accuracy(train)}")
print(f"{dataset.name} {dataset.datatype} Validation Accuracy ({model.name}): {model.accuracy(val)}")


### Cross-Validation with Logistic Regression Tests ###
model = LogisticRegression(verbose=False, normalize_rows=False)
ps = [
    CrossValidationParameter("lr", 1e-10, 0.1, linearSearch=False),
    CrossValidationParameter("lmb", 1e-10, 0.1, linearSearch=False),
    CrossValidationParameter("num_iters", 500)
]
CVParams = model.crossValidateSearch(train, ps)


### LogisticRegression Tests ###
model = LogisticRegression(verbose=False, normalize_rows=False)
model.train(train, lr=CVParams["lr"], lmb=CVParams["lmb"], num_iters=1000)
print(f"{dataset.name} {dataset.datatype} Training Accuracy ({model.name}): {model.accuracy(train)}")
print(f"{dataset.name} {dataset.datatype} Validation Accuracy ({model.name}): {model.accuracy(val)}")


### Cross-Validation with DecisionTree Tests ###
model = DecisionTree()
ps = [
    CrossValidationParameter("maxDepth", 1, 100, linearSearch=False, absolute_max=float("inf")),
    CrossValidationParameter("nodeSize", 1, 100, linearSearch=False, absolute_max=float("inf")),
]
CVParams = model.crossValidateSearch(train, ps, costDelta=0.001, num_folds=2)


### DecisionTree Tests ###
model = DecisionTree()
model.train(train, maxDepth=CVParams["maxDepth"], nodeSize=CVParams["maxDepth"])
print(f"{dataset.name} {dataset.datatype} Training Accuracy ({model.name}): {model.accuracy(train)}")
print(f"{dataset.name} {dataset.datatype} Validation Accuracy ({model.name}): {model.accuracy(val)}")


### DecisionTree Tests ###
model = RandomForest(verbose=False)
model.train(train, maxDepth=27, nodeSize=1, num_trees=2)
print(f"{dataset.name} {dataset.datatype} Training Accuracy ({model.name}): {model.accuracy(train)}")
print(f"{dataset.name} {dataset.datatype} Validation Accuracy ({model.name}): {model.accuracy(val)}")
