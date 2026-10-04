"""DCASE ASD AUC aggregation, cross-checked with a separate
rank/ROC implementation.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def harmonic(values):
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("Invalid metric values")
    return float(len(x) / np.sum(1 / np.maximum(x, np.finfo(np.float64).eps)))


def pairwise_auc(labels, scores):
    a, b = scores[labels == 1], scores[labels == 0]
    if not len(a) or not len(b):
        raise ValueError("Both classes required")
    return float(np.mean((a[:, None] > b) + 0.5 * (a[:, None] == b)))


def independent_pauc(labels, scores, cutoff=0.1):
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    s = scores[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = ends + 1 - tp
    tpr = np.r_[0.0, tp / y.sum()]
    fpr = np.r_[0.0, fp / (1 - y).sum()]
    stop = int(np.searchsorted(fpr, cutoff, side="right"))
    end = np.interp(cutoff, fpr[stop - 1 : stop + 1], tpr[stop - 1 : stop + 1])
    area = np.trapezoid(np.r_[tpr[:stop], end], np.r_[fpr[:stop], cutoff])
    chance = 0.5 * cutoff**2
    return float(0.5 * (1 + (area - chance) / (cutoff - chance)))


def components(labels, domains, scores):
    y = np.asarray(labels, dtype=int)
    domain = np.asarray(domains)
    s = np.asarray(scores, dtype=float)
    if (
        not (len(y) == len(domain) == len(s))
        or set(np.unique(y)) != {0, 1}
        or set(domain) - {"source", "target"}
        or not np.isfinite(s).all()
    ):
        raise ValueError("Invalid evaluation inputs")
    masks = [(domain == "source") | (y == 1), (domain == "target") | (y == 1)]
    direct = [pairwise_auc(y[m], s[m]) for m in masks] + [
        independent_pauc(y, s)
    ]
    official = [roc_auc_score(y[m], s[m]) for m in masks] + [
        roc_auc_score(y, s, max_fpr=0.1)
    ]
    error = max(abs(a - b) for a, b in zip(direct, official))
    if error > 1e-10:
        raise AssertionError("ROC implementations disagree")
    return direct, error
