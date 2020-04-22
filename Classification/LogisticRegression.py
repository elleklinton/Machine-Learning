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