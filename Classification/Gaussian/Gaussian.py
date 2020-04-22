"""
Gaussian Discriminant Analysis (GDA) Python implementation.
For use in classification problems.
Use QDA/LDA to classify, which implement this class.
Do not use this class directly.
Copyright Ellek Linton 2020.
"""

import numpy as np
from Models import ClassificationModel


class Gaussian(ClassificationModel):
    """Pass in NON-Normalized data ONLY for ALL CLASS FUNCTIONS!"""

    def __init__(self, verbose=True, normalize_columns=True, normalize_rows=True):
        super().__init__(verbose, normalize_columns, normalize_rows)
        self.x_dim = None
        self.class_distributions = {}
        self.kludge = None
        self.trained = False

    def __fitClassDistributions__(self, X, y):
        classDistributions = {}
        for label in y:
            mean, cov, count = self.__maximumLiklihoodEstimation__(X, y, label)
            classDict = classDistributions.get(label, {})
            classDict["mean"] = mean
            classDict["cov"] = cov
            classDict["count"] = count
            classDistributions[label] = classDict
        classDistributions["total_count"] = len(X)
        return classDistributions

    def __maximumLiklihoodEstimation__(self, X, y, label):
        if self.verbose: print(f"Calculating mean and sigma for {label}")
        idx = y == label
        data = X[idx]
        means = np.sum(data, axis=0) / len(data)
        cov = (data - means).T @ (data - means)
        cov = cov / data.shape[0]
        return means, cov, len(data)

    def __pretrain__(self, data: np.ndarray, kludge=1e-10):
        """Pass in non-normalized data. Data will be transformed and normalized
        as specified during init automatically."""
        data = self.__shuffle__(data)
        X = self.__preprocess__(data[:, :-1])
        y = data[:, -1]
        self.classes = set(y)
        self.class_distributions = self.__fitClassDistributions__(X, y)
        self.kludge = kludge
        self.trained = True
        return X, y

    def __predict__(self, data : np.ndarray, discriminantFn : lambda x,y:x):
        if data.shape[1] != self.x_dim:
            data = data[:,:-1]
        assert data.shape[1] == self.x_dim, f"Data must have {self.x_dim} columns, but {data.shape[1]} were passed in."

        data = self.__preprocess__(data)
        discValues = np.zeros(shape=(data.shape[0], len(self.classes)))
        for i in self.classes:
            value = discriminantFn(data, i)
            discValues[:, int(i)] = value
        return np.argmax(discValues, axis=1)

    def frequencyOfClass(self, label):
        return self.class_distributions[label]["count"] / self.class_distributions["total_count"]

