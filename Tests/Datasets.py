from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from abc import ABC
from Models import Datatypes

class Data(ABC):
    """
    Abstract class representing a dataset to test Machine Learning models on.
    Data labels are the last column in train/val.
    Train/Val split is 80-20.
    """

    datatype : int
    train : np.ndarray
    val : np.ndarray

class Iris(Data):
    def __init__(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        y = y.reshape((len(y), 1))
        data = np.hstack((X, y))
        train, val = train_test_split(data, train_size=0.8)

        Iris.train = train
        Iris.val = val
        self.datatype = "Classification"
        self.name = "Iris"


class Wine(Data):
    def __init__(self):
        wine = datasets.load_wine()
        X = wine.data
        y = wine.target
        y = y.reshape((len(y), 1))
        data = np.hstack((X, y))
        train, val = train_test_split(data, train_size=0.8)

        Wine.train = train
        Wine.val = val
        self.datatype = "Classification"
        self.name = "Wine"


class Digits(Data):
    def __init__(self):
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target
        y = y.reshape((len(y), 1))
        data = np.hstack((X, y))
        train, val = train_test_split(data, train_size=0.8)

        Digits.train = train
        Digits.val = val
        self.datatype = "Classification"
        self.name = "Digits"

class BreastCancer(Data):
    def __init__(self):
        breastCancer = datasets.load_breast_cancer()
        X = breastCancer.data
        y = breastCancer.target
        y = y.reshape((len(y), 1))
        data = np.hstack((X, y))
        train, val = train_test_split(data, train_size=0.8)

        BreastCancer.train = train
        BreastCancer.val = val
        self.datatype = "Classification"
        self.name = "BreastCancer"

