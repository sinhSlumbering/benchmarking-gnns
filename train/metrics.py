import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np


def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE


def accuracy_TU(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores == targets).float().sum().item()
    return acc


def accuracy_MNIST_CIFAR(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores == targets).float().sum().item()
    return acc


def accuracy_CITATION_GRAPH(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores == targets).float().sum().item()
    acc = acc / len(targets)
    return acc


def accuracy_SBM(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax(torch.nn.Softmax(dim=1)(
        scores).cpu().detach().numpy(), axis=1)
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100. * np.sum(pr_classes) / float(nb_classes)
    return acc


def binary_f1_score(scores, targets):

    # Convert to numpy and get shapes
    y_pred_raw = scores.detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()

    # Debug information

    # Check if pred_scores is 2D with a second dimension of 2 (likely class probabilities)
    if len(y_pred_raw.shape) == 2 and y_pred_raw.shape[1] == 2:
        # Take the second column (probability of class 1) and convert to binary predictions
        y_pred = (y_pred_raw[:, 1] > 0.5).astype(int)
    else:
        # Otherwise, use the standard threshold approach
        y_pred = (y_pred_raw > 0).astype(int)

    # Now both should be 1D arrays with binary values
    return f1_score(y_true, y_pred, average='binary', zero_division=0)


def accuracy_VOC(scores, targets):
    scores = scores.detach().argmax(dim=1).cpu()
    targets = targets.cpu().detach().numpy()
    acc = f1_score(scores, targets, average='weighted')
    return acc


def accuracy_WikiCS(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores == targets).float().sum().item()
    acc = acc / len(targets)
    return acc
