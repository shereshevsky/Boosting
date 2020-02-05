import numpy as np
from Booster import Boosting
from data import get_data

train_x, test_x, train_y, test_y  = get_data()

boost = Boosting()
boost.fit(train_x, train_y)

print(np.mean(np.square(boost.predict(test_x) - test_y)))

print(np.mean(np.square(boost.predict(train_x) - train_y)))
