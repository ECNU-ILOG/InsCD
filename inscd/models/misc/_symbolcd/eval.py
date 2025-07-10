import numpy as np
from sklearn import metrics


def accuracy(y_pred, y_true, threshold=0.5, weights=None):
    pred = np.array(y_pred)
    true = np.array(y_true)
    result = np.where(pred > threshold, 1, 0)
    if weights is not None:
        correct = np.sum((true == result) * weights)
        total = np.sum(weights)
        return correct / total
    else:
        return metrics.accuracy_score(true, result)


def area_under_curve(y_pred, y_true):
    pred = np.array(y_pred)
    true = np.array(y_true)
    fpr, tpr, thresholds = metrics.roc_curve(true, pred)
    return metrics.auc(fpr, tpr)


def f1_score(y_pred, y_true, threshold=0.5):
    pred = np.array(y_pred)
    true = np.array(y_true)
    result = np.where(pred >= threshold, 1, 0)
    return metrics.f1_score(true, result)


def loss(y_pred, y_true):
    pred = np.array(y_pred)
    true = np.array(y_true)
    losses = np.abs(pred - true)
    losses /= np.max(losses)
    return losses
