import numpy as np
from Models import ClassificationModel, TrainingLog
from scipy.special import expit


class LogisticRegression(ClassificationModel):
    def __init__(self, verbose=True, normalize_columns=True, normalize_rows=True, addBias=True):
        super().__init__(verbose, normalize_columns, normalize_rows, addBias)
        self.name = "LogisticRegression"
        self.addBias = addBias
        self.log = TrainingLog()
        self.verbose = verbose

    def __pretrain__(self, data : np.ndarray):
        data = self.__shuffle__(data)
        self.x_dim = data.shape[1] - 1
        X = self.__preprocess__(data[:,:-1])
        y = data[:,-1]
        assert len(set(y)) == 2, f"Logistic regression only works with 2 classes. Got {len(set(y))} classes."
        self.w = np.zeros(X.shape[1])
        return X, y

    def train(self, data, optimizer="batch", batchSize=50, lr=0.0001, lmb=1e-10, num_iters=10000, logFrequency=50):
        """
        :param data: NON normalized data, with the label as the last column
        :param optimizer: Either batch, stochastic, decay.
        :param batchSize: The batch size for each iteration of gradient descent. Only used if using stochastic descent.
        :param lr: Learning rate for gradient descent
        :param lmb: lambda value for regularization. For no regularization, set lmb to 0
        :param num_iters: How many iterations of gradient descent to perform
        :param logFrequency: How frequently to log training progress and print if self.verbose is True
        """
        assert optimizer in ["stochastic", "decay", "batch"], f"Only batch and stochastic gradient descent are allowed! Got: {optimizer}"
        X, y = self.__pretrain__(data)

        logInterval = int(num_iters / logFrequency)
        if logInterval == 0: logInterval = 1

        gradientFn = self.__batchGradient__
        if optimizer in ["stochastic", "decay"]:
            gradientFn = lambda w, lr, lmb, x, y: self.__stochasticGradient__(w, lr, lmb, x, y, batchSize)

        for i in range(num_iters):
            lr_i = lr

            if optimizer == "decay":
                lr_i = (lr_i * num_iters) / (i + 1)

            gradient = gradientFn(self.w, lr_i, lmb, X, y)
            self.w += gradient

            if i % logInterval == 0:
                cost = self.cost(data)
                self.log.iteration.append(i)
                self.log.cost.append(cost)
                self.log.accuracy.append(self.accuracy(data))
                if self.verbose:
                    s = f"Cost at iteration {i}: {round(self.log.cost[-1], 4)}"
                    s += f" (training accuracy: {round(self.log.accuracy[-1], 4)})"
                    print(s)



    # def train(self, optimizer="batch", eps=0.001, lmb=0.1, num_iters=10000, verbose=True, plot_acc=True,
    #           plot_title="Cost vs Number of Iterations"):
    #     """Trains the model on the specified optimizer and returns the final cost after num_iters iterations."""
    #     # assert optimizer in ["batch", "stochastic",
    #     #                      "decay"], f"Only batch and stochastic gradient descent are allowed! Got: {optimizer}"
    #     # self.w = np.zeros(self.x_dim + 1)
    #     # x = self.normalized_training_data[:, :-1]
    #     # y = self.normalized_training_data[:, -1]
    #
    #     # print out progress 10 times:
    #     # print_interval = int(num_iters / 10)
    #     # if print_interval == 0: print_interval = 1
    #
    #     # calculate scoring at 50 evenly spaced intervals
    #     # scoring_interval = int(num_iters / 50)
    #     # if scoring_interval == 0: scoring_interval = 1
    #     # scoringIters = []
    #     # scoringCosts = []
    #     # scoringAccs = []
    #
    #     gradientFunc = self.batchGradient
    #     if optimizer in ["stochastic", "decay"]:
    #         gradientFunc = self.stochasticGradient
    #     for i in range(num_iters):
    #         epsAti = eps
    #         if optimizer == "decay":
    #             epsAti = (eps * num_iters) / (i + 1)
    #         gradient = gradientFunc(self.w, epsAti, lmb, x, y)
    #         self.w += gradient
    #         if verbose and i % print_interval == 0 and i != 0:
    #             s = f"Cost at iteration {i}: {self.cost(self.w, lmb, x, y)}"
    #             s += f" (training accuracy: {round(self.accuracy(self.raw_training_data), 4)})"
    #             print(s)
    #         if i % scoring_interval == 0:
    #             scoringIters.append(i)
    #             scoringCosts.append(self.cost(self.w, lmb, x, y))
    #             scoringAccs.append(self.accuracy(self.raw_training_data))
    #     if verbose: self.plotAccuracy(scoringIters, scoringCosts, scoringAccs, plot_acc, plot_title)
    #     return self.cost(self.w, lmb, x, y)


    # def __init__(self, training_data, train_count=-1):
    #     """Pass in non-normalized data without the bias column for all functions"""
    #     if train_count == -1: train_count = training_data.shape[0]
    #     training_data = training_data[np.random.choice(training_data.shape[0], train_count, replace=False), :]
    #     self.x_dim = training_data.shape[1] - 1
    #     self.featureNorms = None
    #     self.raw_training_data = training_data.copy()
    #     self.normalized_training_data = self.normalize(training_data)
    #     self.w = np.zeros(self.x_dim + 1)
    #
    # def normalize(self, data, addBias=True):
    #     """normalizes columns and optionally adds bias at beginning of design matrix"""
    #     new = self.normalizeColumns(data)
    #     new = self.normalizeRows(new)
    #     if addBias:
    #         if np.average(new[:, 0]) != 1:  # make sure bias has not already been added!
    #             new = np.hstack((np.ones((len(new), 1)), new))
    #     return new
    #
    # def normalizeRows(self, data):
    #     new = data.copy()
    #     if data.shape[1] == self.x_dim + 1:
    #         norm = np.linalg.norm(new[:, :-1], axis=1)
    #         norm[norm == 0] = 1
    #         new[:, :-1] = (new[:, :-1].T / norm).T
    #         return new
    #     else:  # data does not contain label column
    #         norm = np.linalg.norm(new, axis=1)
    #         norm[norm == 0] = 1
    #         new = (new.T / norm).T
    #         return new
    #
    # def normalizeColumns(self, data):
    #     new = data.copy()
    #     if self.featureNorms is None:
    #         self.featureNorms = np.linalg.norm(new[:, :-1], axis=0).reshape((self.x_dim, 1))
    #         self.featureNorms[self.featureNorms == 0] = 1
    #     if new.shape[1] == self.x_dim + 1:
    #         new[:, :-1] = (new[:, :-1].T / self.featureNorms).T
    #         return new
    #     else:  # data does not contain label column
    #         new = (new.T / self.featureNorms).T
    #         return new
    #
    #
    # def plotAccuracy(self, iters, costs, accs, plot_acc, title, x_label="Iteration Number", y_label="Training Accuracy",
    #                  x_scale="linear"):
    #     # referenced https://matplotlib.org/gallery/api/two_scales.html
    #     fig, ax1 = plt.subplots()
    #     ax1.set_xlabel(x_label)
    #     ax1.set_ylabel('Cost', color="tab:blue")
    #     plt.plot(iters, costs, color="tab:blue", label="Cost")
    #     ax1.tick_params(axis='y', labelcolor="tab:blue")
    #     if plot_acc:
    #         ax2 = ax1.twinx()
    #         ax2.set_ylabel(y_label, color="tab:green")
    #         ax2.plot(iters, accs, color="tab:green", label=y_label)
    #         ax2.tick_params(axis='y', labelcolor="tab:green")
    #         ax2.set_xscale(x_scale)
    #     fig.tight_layout()
    #     plt.title(title)
    #     ax1.set_xscale(x_scale)
    #     plt.show()
    #
    # def testBestCutoff(self, data, cutoffs_to_test=np.linspace(0, 1, 1000)):
    #     best_c = 0
    #     best_acc = 0
    #     for c in cutoffs_to_test:
    #         c_acc = self.accuracy(data, c)
    #         if c_acc > best_acc:
    #             best_c = c
    #             best_acc = c_acc
    #     print(f"The cutoff that maximizes accuracy is: {best_c} (accuracy: {best_acc})")
    #     return best_c
    #
    # def accuracy(self, data, cutoff=0.5):
    #     "Pass in UN normalized data as X!"
    #     data = self.normalize(data)
    #     x, y = data[:, :-1], data[:, -1]
    #     pred = self.sigmoid(x, self.w)
    #     labels = pred > cutoff
    #     return np.average(labels == y)
    #     err = np.linalg.norm(y - self.sigmoid(x, self.w))
    #     return err
    #
    def predict(self, data, cutoff=0.5):
        if data.shape[1] != self.x_dim:
            data = data[:,:-1]
        assert data.shape[1] == self.x_dim, f"Data must have {self.x_dim} columns, but {data.shape[1]} were passed in."
        data = self.__preprocess__(data)
        pred = self.sigmoid(data, self.w)
        labels = pred > cutoff
        return labels.astype(int)

    def sigmoid(self, x, w):
        return np.clip(expit(x @ w), 1e-15, 1 - 1e-15)

    def cost(self, data):
        x = self.__preprocess__(data[:,:-1])
        y = data[:,-1]
        pred = self.sigmoid(x, self.w)
        res = y * np.log(pred) + (1 - y) * np.log(1 - pred)
        return -np.average(res)

    def __batchGradient__(self, w, eps, lmb, x, y):
        return -(eps * ((lmb * w) - x.T @ (y - self.sigmoid(x, w))))

    def __stochasticGradient__(self, w, eps, lmb, x, y, batchSize=1):
        idx = np.random.choice(len(x), len(x), replace=False)
        idx = idx[:batchSize]
        randomX = x[idx, :]
        randomY = y[idx]
        return self.__batchGradient__(w, eps, lmb, randomX, randomY)
    #

    #
    # def getFold(self, data, fold):
    #     mask = np.ones(len(data), np.bool)
    #     mask[fold] = 0
    #     val = data[fold]
    #     train = data[mask]
    #     return train, val
    #
    # def crossValidate(self, eps=[10 ** i for i in range(-10, 1)], lmb=0.01, optimizer="batch", num_folds=5,
    #                   num_iters=10000):
    #     """Cross validate a parameter for the model.
    #     You must specify a set eps or lmb, and set the other one to None.
    #     The parameter set to None will be tested.
    #     """
    #     assert isinstance(eps, Iterable) or isinstance(lmb,
    #                                                    Iterable), "You must specify a list of values for either epsilon or lambda!"
    #     assert not (isinstance(eps, Iterable) and isinstance(lmb,
    #                                                          Iterable)), "You cannot set both epsilon and lambda for cross validation!"
    #     param = "epsilon"
    #     if not isinstance(eps, Iterable):
    #         eps = [eps] * len(lmb)
    #         param = "lambda"
    #     else:
    #         lmb = [lmb] * len(eps)
    #         if optimizer == "decay":
    #             param = "theta"
    #     data = self.normalized_training_data
    #     folds = self.calculateFolds(data, num_folds)
    #     all_costs = []
    #     all_accs = []
    #     best_cost = float("inf")
    #     best_acc = float("inf")
    #     best_eps = float("inf")
    #     best_lmb = float("inf")
    #
    #     def params(eps, lmb):
    #         if param in ["epsilon", "theta"]:
    #             if param == "theta":
    #                 return f"[{param}={num_iters}*{eps}]"
    #             return f"[{param}={eps}]"
    #         return f"[lambda={lmb}]"
    #
    #     for eps_l, lmb_l in zip(eps, lmb):
    #         costs = []
    #         accs = []
    #         for fold in folds:
    #             train, val = self.getFold(data, fold)
    #             classifier = LogisticClassifier(train)
    #             train_cost = classifier.train(optimizer, eps_l, lmb_l, num_iters, verbose=False)
    #             val_n = classifier.normalize(val)
    #             cost = classifier.cost(classifier.w, 0, val_n[:, :-1], val_n[:, -1])
    #             val_acc = classifier.accuracy(val)
    #             costs.append(cost)
    #             accs.append(val_acc)
    #         avgCost = np.average(costs)
    #         avgAcc = np.average(accs)
    #         all_costs.append(avgCost)
    #         all_accs.append(avgAcc)
    #         print(f"{params(eps_l, lmb_l)} Cross-validated cost: {avgCost} (accuracy: {round(avgAcc, 4)})")
    #         if avgCost < best_cost:
    #             best_cost = avgCost
    #             best_acc = avgAcc
    #             best_eps = eps_l
    #             best_lmb = lmb_l
    #     print(
    #         f"\nCross-validation complete! The best hyperparameter is: {params(best_eps, best_lmb)} (cost: {round(best_cost, 4)}, accuracy: {round(best_acc, 4)})")
    #     paramToPlot = eps
    #     if param == "lambda":
    #         paramToPlot = lmb
    #     self.plotAccuracy(paramToPlot, all_costs, all_accs, False,
    #                       f"Cross-Validation of {param}",
    #                       param,
    #                       "Validation Accuracy",
    #                       "log")
    #     if param in ["epsilon", "theta"]:
    #         return best_eps
    #     return best_lmb