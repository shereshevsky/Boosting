from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


def get_data():
    boston = load_boston()
    boston_y = boston.target
    boston_X = boston.data
    train_x, test_x, train_y, test_y = train_test_split(boston_X, boston_y)
    return train_x, test_x, train_y, test_y
