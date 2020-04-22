"""
Quadratic Discriminant Analysis (QDA) Python implementation.
For use in classification problems.
Copyright Ellek Linton 2020.
"""

import numpy as np
from Classification.Gaussian.Gaussian import Gaussian

class QDA(Gaussian):
    def __init__(self, verbose=True, normalize_columns=True, normalize_rows=True):
        super().__init__(verbose, normalize_columns, normalize_rows)
        self.name = "QDA"

    def train(self, data: np.ndarray, kludge=1e-10):
        """Pass in non-normalized data. Data will be transformed and normalized
        as specified during initialization automatically."""
        X, y = super().__pretrain__(data, kludge)

    def predict(self, data : np.ndarray):
        """Pass in NON-NORMALIZED data!"""
        assert self.trained, "You must run QDA.train() before running QDA.predict()."

        return self.__predict__(data, self.__quadraticDiscriminant__)

    def __quadraticDiscriminant__(self, data, label):
        assert label in self.classes, f"{label} is not a valid class. Valid classes are: {self.classes}"

        u = self.class_distributions[label]["mean"]
        sigma = self.class_distributions[label]["cov"]
        sigma = sigma.copy() + self.kludge * np.identity(sigma.shape[0])
        x_centered = (data - u)
        sigmaInvXCentered = np.linalg.solve(sigma, x_centered.T)
        res = []
        for i in range(len(data)):
            res.append(x_centered[i, :] @ sigmaInvXCentered[:, i])
        res = np.array(res) * -1 / 2

        res -= 1 / 2 * np.linalg.slogdet(sigma)[1]
        res += np.log(self.frequencyOfClass(label))
        return res  # np.diag(res)