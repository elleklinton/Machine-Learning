"""
Linear Discriminant Analysis (LDA) Python implementation.
For use in classification problems.
Copyright Ellek Linton 2020.
"""

import numpy as np
from Classification.Gaussian.Gaussian import Gaussian

class LDA(Gaussian):
    def __init__(self, verbose=True, normalize_columns=True, normalize_rows=True):
        super().__init__(verbose, normalize_columns, normalize_rows)
        self.name = "LDA"

    def train(self, data: np.ndarray, kludge=1e-10):
        """Pass in non-normalized data. Data will be transformed and normalized
        as specified during initialization automatically."""
        X, y = super().__pretrain__(data, kludge)
        self.sigma = self.__pooledClassCovariance__()

    def predict(self, data : np.ndarray):
        """Pass in NON-NORMALIZED data!"""
        assert self.trained, "You must run LDA.train() before running LDA.predict()."

        return self.__predict__(data, self.__linearDiscriminant__)

    def __linearDiscriminant__(self, data, label):
        assert label in self.classes, f"{label} is not a valid class. Valid classes are: {self.classes}"

        sigma = self.sigma.copy() + self.kludge * np.identity(self.sigma.shape[0])
        u = self.class_distributions[label]["mean"]
        sigmaInvX = np.linalg.solve(sigma, data.T)
        sigmaInvU = np.linalg.solve(sigma, u)
        res = u.T @ sigmaInvX
        res -= (1 / 2) * (u.T @ sigmaInvU)
        res += np.log(self.frequencyOfClass(label))
        return res

    def __pooledClassCovariance__(self):
        sigma = np.zeros(shape=(self.x_dim, self.x_dim))
        for classDict in self.class_distributions.values():
            if isinstance(classDict, dict):
                sigma += classDict["cov"]
        sigma = sigma / len(self.class_distributions.values())
        return sigma