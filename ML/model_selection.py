import numpy as np
from itertools import product
from .metrics import r2_score


def train_test_split(X,y,test_ratio=0.2,seed = None):
    '''
    :param X: input data set
    :param y: input lable set
    :param test_ratio: the proportion of test data set
    :param seed: random seed
    :return: X_train,y_train,X_test,y_test
    '''

    assert X.shape[0] == y.shape[0], \
        "the len of the X must be equal to the len the y"
    assert 0 <= test_ratio <= 1, \
        "test_ratio must be more than 0 and less than 1"
    if seed:
        np.random.seed(seed)

    # 打乱数据
    shuffle_indexes = np.random.permutation(len(X))
    test_number = int(len(X) * test_ratio)
    X_test = X[shuffle_indexes[:test_number], :]
    y_test = y[shuffle_indexes[:test_number]]
    X_train = X[shuffle_indexes[test_number:], :]
    y_train = y[shuffle_indexes[test_number:]]

    return X_train, X_test, y_train, y_test


class GridSearchCV():
    def __init__(self, estimator, Parameters):
        self.estimator = estimator
        self.Parameters = Parameters
        self.score_ = None
        self.bestPara_ = None

    def get_gridPara(self):
        gridPara = []
        for p in self.Parameters:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                pass
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    gridPara.append(params)
        return  gridPara

    def fit_getbestScore(self,X_train,y_train,X_test,y_test):
        self.score_ = 0
        gridPara = self.get_gridPara()
        for para in gridPara:
            for key,value in para.items():
                setattr(self.estimator, key, value)
            self.estimator.fit(X_train,y_train)
            # y_predict =self.estimator.predict(X_test)
            # score = r2_score(y_test, y_predict)
            score = self.estimator.score(X_test, y_test)
            if score > self.score_:
                self.score_ = score
                self.bestPara_ = para
        return self
