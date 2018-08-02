import numpy as np

class PCA:

    def __init__(self,n_components):
        """初始化PCA"""
        self.n_components = n_components
        self.components_ = None
    def demean(self,X):
        return X - np.mean(X, axis=0)

    def fit(self,X,eta=0.001, n_iters=1e4, epsilon=1e-8):
        '''
        X:为样本矩阵
        eta : 学习速率
        n_iters；迭代的最大次数
        epsilon : 残差（当前函数值与上次函数值之间的差值）的最小值
        '''

        def f(X, w):
            '''
            w：为样本所要映射的方向向量
            '''
            return (X.dot(w)).T.dot(X).dot(w) / len(X)

        def df(X, w):
            return X.T.dot(X).dot(w) * 2. / len(X)

        # 注意 我们假设w是单位的方向向量，但是，在每次迭代更新w时，并不能保证w一直是单位向量，故需要将w单位化
        def unit_w(w):
            return w / np.linalg.norm(w)

        def first_component(X, init_w):
            '''
            init_w: 初始化的方向向量
            '''
            w = init_w
            w = unit_w(w)
            i_iters = 0
            while i_iters < n_iters:
                i_iters += 1
                last_w = w
                w = w + eta * df(X, w)
                w = unit_w(w)
                if f(X, w) - f(X, last_w) <= epsilon:
                    break

            return w

        X = self.demean(X)
        self.components_ = np.empty(shape=(self.n_components,X.shape[1]))
        init_w = np.random.random(size=X.shape[1])
        for i in range(self.n_components):
            w = first_component(X, init_w)
            self.components_[i,:] = wzs
            X = X - X.dot(w).reshape(-1, 1) * w
        return self

    def transform(self,X):
        '''
        将给定的样本矩阵X映射到各个主成分分量中
        :param X:m*n 原始的样本矩阵
        :return:Xk m*k 降维后的样本矩阵
        '''
        return X.dot(self.components_.T)

    def inverse_transform(self,Xk):
        '''
        将低维的数据再映射回高维空间
        但注意，即使恢复到原始的维度，但是不再是原始的样本矩阵,因为在降维的过程中丢失了部分信息
        :param Xk:降维后的样本矩阵
        :return:高维空间(原始维数)矩阵
        '''
        return Xk.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
