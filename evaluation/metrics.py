import torch
import numpy as np

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def precision(y_true, y_pred, num_classes=10):
    precisions = []

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))

        if tp + fp == 0:
            precisions.append(0)
        else:
            precisions.append(tp / (tp + fp))

    return np.mean(precisions)

def recall(y_true, y_pred, num_classes=10):
    recalls = []

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))

        if tp + fn == 0:
            recalls.append(0)
        else:
            recalls.append(tp / (tp + fn))

    return np.mean(recalls)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        return 0

    return 2 * (p * r) / (p + r)