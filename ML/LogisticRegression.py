import numpy as np
# 在LogisticRegression中，我们预测的是类别，所以需要用
from .metrics import accuracy_score
class LogisticRegression():
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None       #__thet:Private variables

    def _sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def fit(self, X, y, eta=0.01, iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X.shape[0] == y.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(Xb, y, theta):
            y_hat = self._sigmoid(Xb.dot(theta))
            try:
                return -(y.dot(np.log(y_hat))+ (1-y).dot(np.log(1-y_hat)))/ len(y)
            except:
                return float('inf')

        def dJ(Xb, y, theta):
            y_hat = self._sigmoid(Xb.dot(theta))
            return Xb.T.dot(y_hat - y)  / len(y)

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
        self._theta = gradient_descent(Xb, y, theta)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self



    def predict_prob(self,X):
        assert self.coef_ is not None and self.interception_ is not None, \
            "must fit before predict"
        assert X.shape[1] == len(self.coef_), \
            "the mumber of fetures must be equal to the len of coef_"
        Xb = np.hstack((np.ones((X.shape[0], 1)), X))

        return self._sigmoid(Xb.dot(self._theta))

    def predict(self,X):
        y_predict_prob = self.predict_prob(X)
        return np.array(y_predict_prob >= 0.5, dtype = 'int')

    def score(self,X,y):
        y_predict = self.predict(X)
        return accuracy_score(y,y_predict)

    def __repr__(self):
        return "LogisticRegression()"