import numpy as np
from sklearn.tree import DecisionTreeRegressor
from collections import defaultdict


class Boosting:
    """
    Use the scikit-learn's DecisionTreeRegressor with `max_depth = 1` (stumps)
     to write a L2Boost model which minimize the L2 square loss iteration by iteration.
    Reminder: in each step, build a decision tree to minimize the error between the true label and the accumulated (sum)
    of the previous step predictions.
    """

    def __init__(self, n_trees=1, n_steps=3, l_rate=0.1):
        self.n_trees = n_trees
        self.n_steps = n_steps
        self.l_rate = l_rate

        self.predictors = []
        self.residuals = []

    def fit(self, x, y):
        f = MeanPredict()
        f.fit(y)
        self.predictors.append(f)
        res = y - f.predict(x)  # residuals direction vector
        self.residuals.append(res)

        for m in range(self.n_steps):
            dm = DecisionTreeRegressor(criterion='mse').fit(x, res)
            self.predictors.append(dm)
            res = + self.l_rate * dm.predict(x)
            self.residuals.append(res)

    def predict(self, x):
        return np.array([sum([predictor.predict(i.reshape(1, -1)) for predictor in self.predictors])[0]
                for i in x])


class MeanPredict:
    def fit(self, y):
        self.prediction = np.mean(y)

    def predict(self, x):
        return np.array([self.prediction for _ in range(x.shape[0])])
