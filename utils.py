"""Evaluation metrics."""

import math
import numpy as np

def nDCG_Time(ground_truth, _recList):
    rec_num = len(_recList)
    
    idealOrder = ground_truth
    idealDCG = 0.0
    for j in range(min(rec_num, len(idealOrder))):
        idealDCG += ((math.pow(2.0, len(idealOrder) - j) - 1) / math.log(2.0 + j))

    recDCG = 0.0
    for j in range(rec_num):
        item = _recList[j]
        if item in ground_truth:
            rank = len(ground_truth) - ground_truth.index(item)
            recDCG += ((math.pow(2.0, rank) - 1) / math.log(1.0 + j + 1))

    return (recDCG / idealDCG)


def Recall(_test_set, _recList):
    hit = len(set(_recList).intersection(set(_test_set)))
    return hit / float(len(_test_set))


def Precision(_test_set, _recList):
    hit = len(set(_recList).intersection(set(_test_set)))
    return hit / float(len(_recList))

def AveragePrecisionAtK(_test_set, _recList, k):
    if not _test_set:
        return 0.0
    
    if len(_recList) > k:
        _recList = _recList[:k]
    
    score = 0.0
    hits = 0.0

    for i, p in enumerate(_recList):
        if p in _test_set and p not in _recList[:i]:
            hits += 1.0
            score += hits / (i + 1.0)
    
    return score / min(len(_test_set), k)

def MeanAveragePrecisionAtK(_test_sets, _recLists, k):
    """
    Computes the mean average precision at k.

    Parameters
    ----------
    _test_sets : list of lists of items that are to be predicted for users

    _recLists : list of lists of recommended items for users

    Returns
    -------
    score : float, the mean average precision at k over the input lists
    """

    return np.mean([AveragePrecisionAtK(t, r, k) for t, r in zip(_test_sets, _recLists)])
