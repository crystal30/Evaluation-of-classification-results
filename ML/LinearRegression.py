import numpy as np
import math
from .metrics import r2_score
class LinearRegression():
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self.__thet = None       #__thet:Private variables

    def fit(self,X,y):
        assert len(X) == len(y), \
            "the length of x must be equal to the length of y"
        Xb = np.hstack((np.ones((X.shape[0], 1)), X))
        self.__thet = np.linalg.inv(Xb.T.dot(Xb)).dot(Xb.T).dot(y)
        self.interception_ = self.__thet[0]
        self.coef_ = self.__thet[1:]
        return self

    def fit_gd(self, X, y, eta=0.01, iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X.shape[0] == y.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(Xb, y, theta):
            try:
                return (y - Xb.dot(theta)).T.dot(y - Xb.dot(theta)) / len(y)
            except:
                return float('inf')

        def dJ(Xb, y, theta):
            return Xb.T.dot(Xb.dot(theta) - y) * 2 / len(y)

        def gradient_descent(Xb, y, init_theta, epsilon=1e-8):
            theta = init_theta
            i_iters = 0
            while i_iters < iters:
                i_iters += 1
                last_theta = theta
                theta = theta - eta * dJ(Xb, y, theta)
                if abs(J(Xb, y, theta) - J(Xb, y, last_theta)) < epsilon:
                    break
            return theta

        Xb = np.hstack([np.ones((len(X), 1)), X])
        # note: init_theta = np.zeros(shape= (Xb.shape[1],1)) is wrong;
        #because y.ndim is 1, so init_theta.ndim should be 1;
        ##矩阵和一维的向量（默认均为列向量,其ndim = 1）相乘 == 矩阵和二维的矩阵（只有一列,但ndim=2）相乘;
        #向量（默认为列向量）a - 向量（默认为列向量）b 得到的仍是一维的列向量;
        #向量（默认为列向量）a - 矩阵（只有一列）B 得到的是 二维的矩阵
        #eg：a.shape=(10,);B.shape=(10,1);(a-B).shape=(10,10)

        theta = np.zeros(Xb.shape[1]) #
        self.__thet = gradient_descent(Xb, y, theta)

        self.interception_ = self.__thet[0]
        self.coef_ = self.__thet[1:]

        return self

    def fit_sgd(self, X, y, iters=5,t0=5, t1=50):

        assert X.shape[0] == y.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def dJ_sgd(Xb_i, y_i, theta):
            return Xb_i.T.dot(Xb_i.dot(theta) - y_i) * 2

        def eta(t):
            return t0 / (t + t1)

        def sgd(Xb, y, init_theta):
            theta = init_theta
            i_iters = 0
            while i_iters < iters:
                index = np.random.permutation(len(y))
                ti = 0
                for i in index:
                    theta = theta - eta(i_iters * len(y) + ti) * dJ_sgd(Xb[i, :], y[i], theta)
                    ti += 1
                i_iters += 1
            return theta

        Xb = np.hstack([np.ones((len(X), 1)), X])
        init_theta = np.zeros(Xb.shape[1])
        self.__thet = sgd(Xb, y, init_theta)
        self.interception_ = self.__thet[0]
        self.coef_ = self.__thet[1:]

        return self

    def fit_minigd(self, X, y, bach_sample_ratio=0.1, iters=5,t0 = 5,t1 = 50):
        assert X.shape[0] == y.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def dJ_minigd(Xb_bach, y_bach, theta):
            return Xb_bach.T.dot(Xb_bach.dot(theta) - y_bach) * 2/len(y_bach)

        def eta(i_iters):
            return t0 / (i_iters + t1)

        def minigd(Xb, y, init_theta):
            theta = init_theta
            i_iters = 0
            bach = math.floor(bach_sample_ratio * len(y))
            while i_iters < iters:
                index = np.random.permutation(len(y))
                ti = 0
                for i in range(1, len(y) // bach + 1):
                    bach_index = index[(i - 1) * bach:i * bach]
                    theta = theta - eta(i_iters * (len(y) // bach) + ti) * dJ_minigd(Xb[bach_index, :], y[bach_index], theta)
                    ti += 1
                i_iters += 1
            return theta

        Xb = np.hstack([np.ones((len(X), 1)), X])
        init_theta = np.zeros(Xb.shape[1])
        self.__thet = minigd(Xb, y, init_theta)
        self.interception_ = self.__thet[0]
        self.coef_ = self.__thet[1:]

        return self

    def predict(self,X):
        assert self.coef_ is not None and self.interception_ is not None, \
            "must fit before predict"
        assert X.shape[1] == len(self.coef_), \
            "the mumber of fetures must be equal to the len of coef_"

        return np.hstack((np.ones((X.shape[0], 1)), X)).dot(self.__thet)


    def score(self,X,y):
        y_predict = self.predict(X)
        return r2_score(y,y_predict)

    def __repr__(self):
        return "LinearRegression()"