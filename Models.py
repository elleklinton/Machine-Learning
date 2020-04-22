from abc import ABC
import numpy as np
import inspect
from collections.abc import Iterable
from typing import List


class Datatypes:
    categorical = 0
    quantitative = 1

class TrainingLog():
    def __init__(self):
        self.accuracy = []
        self.cost = []
        self.iteration = []

class CrossValidationParameter:
    def __init__(self, name : str, windowMinOrDefault : float = None,
                 windowMax : float = None, linearSearch=True, searchCount=5,
                 absolute_min=0, absolute_max=1):
        self.name = name
        self.windowMin = windowMinOrDefault
        self.windowMax = windowMax
        self.linearSearch = linearSearch
        self.searchCount = searchCount
        self.constant = windowMax is None
        self.default = windowMinOrDefault
        self.absolute_min = absolute_min
        self.absolute_max = absolute_max
        if self.windowMax is not None:
            self.default = (self.windowMin + self.windowMax) / 2

    def valuesToTest(self):
        valuesToTry = []
        minBound = np.clip(self.windowMin, self.absolute_min + 1e-15, self.absolute_max)
        maxBound = np.clip(self.windowMax, self.absolute_min + 1e-15, self.absolute_max)
        if self.linearSearch:
            valuesToTry.extend(np.linspace(minBound, maxBound, self.searchCount))
        else:
            valuesToTry.extend(np.geomspace(minBound, maxBound, self.searchCount))
        return valuesToTry

    def updateRange(self, bestValueFound, spaceDecay):
        currRange = self.windowMax - self.windowMin
        newRange = (currRange * spaceDecay) / 2
        self.windowMin = bestValueFound - newRange
        self.windowMax = bestValueFound + newRange
        self.default = (self.windowMin + self.windowMax) / 2


    @staticmethod
    def multiplePayloads(existingPayload, otherParams):
        existingPayload = existingPayload.copy()
        for p in otherParams:
            existingPayload[p.name] = p.default
        return existingPayload

    @staticmethod
    def paramListToStr(params):
        d = {}
        for p in params:
            d[p.name] = p.default
        return ClassificationModel.__dictStr__(ClassificationModel, d)

class ClassificationModel(ABC):

    name = "Model"

    def __init__(self, verbose=True, normalize_columns=True, normalize_rows=True, addBias=False):
        self.__normalization_cache__ = {}
        self.verbose = verbose
        self.normalize_columns = normalize_columns
        self.normalize_rows = normalize_rows
        self.addBias = addBias

    def train(self, data):
        pass

    def accuracy(self, data : np.ndarray):
        pred = self.predict(data[:,:-1])
        truth = data[:,-1]
        return np.average(pred == truth)

    def predict(self, data : np.ndarray):
        pass

    def cost(self, data : np.ndarray):
        return 1 - self.accuracy(data)

    def __preprocess__(self, X: np.ndarray):
        if self.x_dim is None:
            self.x_dim = X.shape[1]

        assert X.shape[1] == self.x_dim, "To preprocess, the label column must be removed."

        if self.normalize_columns:
            X = self.__normalize__(X, axis=0, addBias=self.addBias)
        if self.normalize_rows:
            X = self.__normalize__(X, axis=1, addBias=self.addBias)
        return X

    def __shuffle__(self, data : np.ndarray):
        idx = np.random.choice(data.shape[0], len(data), replace=False)
        data = data[idx, :]
        return data

    def __normalize__(self, data, axis=1, addBias=False):
        """Data must be passed in WITHOUT the labels

        This will divide the specified axis by its l2 norm.
        If axis=0, columns will be normalized
        and the respective norms will be stored so validation/test data can
        be normalized by the same constants.
        If axis=1, rows will be normalized and nothing will be stored because
        row normalization is independent for each row."""
        new = data.copy()
        if axis == 0:
            if not "column_norms" in self.__normalization_cache__.keys():
                norm = np.linalg.norm(new, axis=axis)
                self.__normalization_cache__["column_norms"] = norm
            else:
                norm = self.__normalization_cache__["column_norms"]
        elif axis == 1:
            norm = np.linalg.norm(new, axis=axis)
        else:
            raise NotImplemented("Only axis 0/1 can be normalized.")

        norm[norm == 0] = 1
        if axis == 0:
            new = new / norm.reshape((1, len(norm)))
        else:
            new = (new.T / norm).T

        if addBias:
            new = np.hstack((np.ones((len(new), 1)), new))
        return new

    def __checkCV__(self, params : dict):
        trainParams = inspect.signature(self.train).parameters.keys()
        iterValues = [v for v in params.values() if isinstance(v, Iterable)]
        assert len(iterValues) > 0, "You must iterate over at least one variable for cross-validation."
        valueIterationCount = len(iterValues[0])
        for p in params.keys():
            assert p in trainParams, f"Invalid parameter to cross-validate. Received {p} but must be one of: {trainParams}"
            if isinstance(params[p], Iterable):
                assert len(params[p]) == valueIterationCount, f"Iterated Value lists must all be of the same length. Expected {valueIterationCount} but got {len(params[p])}."
            else:
                params[p] = [params[p]] * valueIterationCount
        return valueIterationCount

    def __getIterParamsCV__(self, data, params : dict, i):
        paramPayload = {}
        for k, v in params.items():
            paramPayload[k] = v[i]
        paramPayload["data"] = data
        return paramPayload

    def __calculateFolds__(self, data, num_folds):
        idxs = np.arange(len(data))
        np.random.shuffle(idxs)
        IDXGroups = np.array_split(idxs, num_folds)
        return np.array(IDXGroups)

    def __getFold__(self, data, fold):
        mask = np.ones(len(data), np.bool)
        mask[fold] = 0
        val = data[fold]
        train = data[mask]
        return train, val

    def __CVStep__(self, paramPayload, folds):
        cost, accuracy = [], []
        for fold in folds:
            train, val = self.__getFold__(paramPayload["data"], fold)
            model = self.__class__(verbose=False, normalize_columns=self.normalize_columns, normalize_rows=self.normalize_rows)
            model.train(**paramPayload)
            val_cost = model.cost(val)
            val_acc = model.accuracy(val)
            cost.append(val_cost)
            accuracy.append(val_acc)
        avgCost = np.average(cost)
        avgAcc = np.average(accuracy)
        return avgCost, avgAcc



    def crossValidate(self, data, params : dict, num_folds = 5, verbose = True):
        """:param params: A dictionary of different values to cross-validate.
        The key of the dictionary must be a parameter in self.train, and the
        value of each key should be a list of parameters to try. For example: {"lr":[0.01, 0.001]}

        :param num_folds: How many folds of cross-validation to perform.

        :returns: The set of parameters yielding the lowest cost"""

        iterations = self.__checkCV__(params)
        folds = self.__calculateFolds__(data, num_folds)
        paramHistory, costHistory, accuracyHistory = [], [], []
        for i in range(iterations):
            paramPayload = self.__getIterParamsCV__(data, params, i)
            CVCost, CVAcc = self.__CVStep__(paramPayload, folds)
            paramHistory.append(paramPayload)
            costHistory.append(CVCost)
            accuracyHistory.append(CVAcc)
            if verbose:
                del paramPayload["data"]
                p = self.__dictStr__(paramPayload)
                print(f"[Cross-Validation with: {{{p}}}] Cost: {round(CVCost, 4)}, Accuracy: {round(CVAcc, 4)}")
        bestIdx = np.argmin(costHistory)
        if verbose:
            p = self.__dictStr__(paramHistory[bestIdx])
            print(f"Cross-Validation complete! The best parameters found are: {{{p}}} Cost: {round(costHistory[bestIdx], 4)}, Accuracy: {round(accuracyHistory[bestIdx], 4)})")
        return paramHistory[bestIdx], costHistory[bestIdx], accuracyHistory[bestIdx]

    def crossValidateSearch(self, data, paramsToSearch=List[CrossValidationParameter], costDelta = 1e-5, spaceDecay = 0.5, momentumIters = 3, num_folds=5, verbose=True):
        constantParams = [p for p in paramsToSearch if p.constant]
        constantPayload = {}
        for p in constantParams: constantPayload[p.name] = p.default

        dynamicParams = [p for p in paramsToSearch if not p.constant]

        lastCost = float("inf")
        i = 0
        noChangeIters = 0

        while True:
            paramToTest : CrossValidationParameter = dynamicParams.pop(0)
            payload = CrossValidationParameter.multiplePayloads(constantPayload, dynamicParams)
            payload[paramToTest.name] = paramToTest.valuesToTest()

            bestParams, bestCost, bestAcc = self.crossValidate(data, payload, num_folds, False)
            paramToTest.updateRange(bestParams[paramToTest.name], spaceDecay)
            dynamicParams.append(paramToTest)
            if verbose:
                ps = CrossValidationParameter.paramListToStr(dynamicParams)
                s = f"Cross-Validation Iteration {i}. Best params so far are:\n" \
                    f"{ps}\n" \
                    f"(cost={round(bestCost, 6)}, acc={round(bestAcc, 6)}, costDelta={round(abs(lastCost - bestCost), 6)})"
                print(s)
            if abs(lastCost - bestCost) < costDelta:
                noChangeIters += 1
                if noChangeIters == momentumIters:
                    del bestParams["data"]
                    ps = self.__dictStr__(bestParams)
                    s = f"Cross-Validation search complete! The best parameters are: {ps}"
                    print(s)
                    return bestParams
            lastCost = bestCost
            i += 1

    def __dictStr__(self, d : dict):
        s = ""
        for k, v in d.items():
            s += f"{k}={v}, "
        return s[:-2]