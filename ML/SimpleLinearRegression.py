import numpy as np
from .metrics import  r2_score

class SimpleLinearRegression_one():

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self,x,y):
        assert x.ndim is 1 and y.ndim is 1,\
            "x,y must be 1 dimensional,the SimpleLinearRegression can only solve singal feature train data"
        assert len(x) == len(y), "the length of x must be equal to the length of y"
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        molecular = 0.0
        denominator = 0.0

        for i in range(len(x)):
            molecular += (x[i] - x_mean) * (y[i] - y_mean)
            denominator += (x[i] - x_mean) ** 2
        self.a_ = molecular / denominator
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self,x):
        assert x.ndim is 1, "y must be 1 dimensional"
        assert self.a_ is not None and self.b_ is not None,\
            "must fit before predict"
        return np.array([self.__predict(xi) for xi in x ])

    def __predict(self,x):

        y_hat = self.a_ * x + self.b_
        return y_hat

    def __repr__(self):
        return "SimpleLinearRegression_one()"

class SimpleLinearRegression_twoa():
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self,x,y):
        assert x.ndim is 1 and y.ndim is 1,\
            "x,y must be 1 dimensional,the SimpleLinearRegression can only solve singal feature train data"
        assert len(x) == len(y), "the length of x must be equal to the length of y"
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        self.a_ = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean) ** 2)
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self,x):
        assert x.ndim is 1, "y must be 1 dimensional"
        assert self.a_ is not None and self.b_ is not None,\
            "must fit before predict"
        return np.array([self.__predict(xi) for xi in x ])

    def __predict(self,x):

        y_hat = self.a_ * x + self.b_
        return y_hat

    def __repr__(self):
        return "SimpleLinearRegression_twoa()"

class SimpleLinearRegression_twob():
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self,x,y):
        assert x.ndim is 1 and y.ndim is 1,\
            "x,y must be 1 dimensional,the SimpleLinearRegression can only solve singal feature train data"
        assert len(x) == len(y), "the length of x must be equal to the length of y"
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        self.a_ = (x - x_mean).dot(y - y_mean) / (x - x_mean).dot(x - x_mean)
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self,x):
        assert x.ndim is 1, "y must be 1 dimensional"
        assert self.a_ is not None and self.b_ is not None,\
            "must fit before predict"
        return np.array([self.__predict(xi) for xi in x ])

    def __predict(self,x):

        y_hat = self.a_ * x + self.b_
        return y_hat

    def score(self,x,y):
        y_predict = self.predict(x)
        return r2_score(y,y_predict)

    def __repr__(self):
        return "SimpleLinearRegression_twob()"

