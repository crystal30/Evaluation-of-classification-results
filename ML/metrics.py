import numpy as np
from math import sqrt


def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y,y_predict):
    assert y.shape == y_predict.shape, \
        "the size of y must be equal to the size of the y_predict"

    return sum((y - y_predict)**2)/len(y)

def root_mean_squared_error(y,y_predict):

    return sqrt(mean_squared_error(y,y_predict))

def mean_absolute_error(y,y_predict):
    assert y.shape == y_predict.shape, \
        "the size of y must be equal to the size of the y_predict"

    return sum(np.absolute(y - y_predict))/len(y)

def r2_score(y,y_predict):

    return 1-(mean_squared_error(y,y_predict)/np.var(y))

# confusion_matrix

def TN(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return sum((y_true == 0) & (y_pred == 0))

def FP(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return sum((y_true == 0) & (y_pred == 1))

def FN(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return sum((y_true == 1) & (y_pred == 0))

def TP(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return sum((y_true == 1) & (y_pred == 1))

def confusion_matrix(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.array([[TN(y_true, y_pred), FP(y_true, y_pred)],
                     [FN(y_true, y_pred), TP(y_true, y_pred)]])


def precision_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    try:
        return TP(y_true, y_pred)/(TP(y_true, y_pred) + FP(y_true, y_pred))
    except:
        return 0

def recall_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    try:
        return TP(y_true, y_pred)/(TP(y_true, y_pred) + FN(y_true, y_pred))
    except:
        return 0

def f1_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    try:
        return 2 * precision_score(y_true, y_pred) * recall_score(y_true, y_pred)/(precision_score(y_true, y_pred) + recall_score(y_true, y_pred))
    except:
        return 0

def fpr(y_true, y_pred):
    myconsusion_matrix = confusion_matrix(y_true, y_pred)
    tn = myconsusion_matrix[0][0]
    fp = myconsusion_matrix[0][1]
    try:
        return fp / (tn+fp)
    except:
        return 0

def tpr(y_true, y_pred):
    myconsusion_matrix = confusion_matrix(y_true, y_pred)
    fn = myconsusion_matrix[1][0]
    tp = myconsusion_matrix[1][1]
    try:
        return tp / (fn+tp)
    except:
        return 0




