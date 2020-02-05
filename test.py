import numpy as np
import matplotlib.pyplot as plt

from Booster import GBTL2
from data import get_data

train_x, test_x, train_y, test_y = get_data()

boost = GBTL2(n_trees=20, l_rate=0.1, max_depth=5)
boost.fit(train_x, train_y)

from collections import defaultdict

res = defaultdict()

n_trees = [3, 6, 12, 24, 48, 100, 300, 500, 1000]
l_rate = [0.01, 0.05, 0.08, 0.1, 0.2, 0.5]
max_depth = [1, 3, 5, 10, 20, 100]

for n in n_trees:
    for l in l_rate:
        for m in max_depth:
            b = GBTL2(n, l, m)
            b.fit(train_x, train_y)
            res[f"{n}_{l}_{m}"] = np.mean(np.square(b.predict(test_x) - test_y))

best_n_trees, best_l_rate, best_max_depth = list(res.keys())[np.argmin(np.array(list(res.values())))].split('_')

b = GBTL2(int(best_n_trees), float(best_l_rate), int(best_max_depth))
b.fit(train_x, train_y)
predicted = b.predict(test_x)

plt.scatter(test_y, predicted)
