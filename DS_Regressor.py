

class DS_Regressor(DecisionTreeRegressor):
    def __init__(self):
        return super().__init__(max_depth=1)
