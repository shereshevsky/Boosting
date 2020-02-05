import numpy as np
from sklearn.tree import DecisionTreeRegressor


class MeanPredict:
    """helper for average value `prediction`"""

    def fit(self, y):
        self.prediction = np.mean(y)

    def predict(self, x):
        return np.array([self.prediction for _ in range(x.shape[0])])


class GBTL2:
    """
    Use the scikit-learn's DecisionTreeRegressor with `max_depth = 1` (stumps)
     to write a L2Boost model which minimize the L2 square loss iteration by iteration.
    Reminder: in each step, build a decision tree to minimize the error between the true label and the accumulated (sum)
    of the previous step predictions.
    """

    def __init__(self, n_trees=5, l_rate=0.05, max_depth=5, eps=0.0001):
        self.n_trees = n_trees
        self.l_rate = l_rate
        self.max_depth = max_depth
        self.eps = eps

        self.predictors = []
        self.residuals = []

    def fit(self, x, y):
        f = MeanPredict()
        f.fit(y)
        self.predictors.append(f)
        res = y - f.predict(x)  # residuals direction vector
        self.residuals.append(res)

        for m in range(self.n_trees):
            dm = DecisionTreeRegressor(criterion='mse', max_depth=self.max_depth).fit(x, res)
            self.predictors.append(dm)
            res = + self.l_rate * dm.predict(x)
            self.residuals.append(res)

            if np.sum(np.square(res)) < self.eps:
                break

    def predict(self, x):
        return np.array([np.sum([predictor.predict(i.reshape(1, -1)) for predictor in self.predictors]) for i in x])

