# From https://github.com/oravus/DeltaDescriptors/blob/master/src/outFuncs.py
# Copyright Sourav Garg


import numpy as np


def getPAt100RFromPRVals(prvals_in):
    try:
        return np.max(prvals_in[prvals_in[:, 1] >= 1.0][:, 0])
    except (ValueError, IndexError):
        return 0.0


def getRAtXPFromPRVals(prvals_in, precision=1.0):
    try:
        return np.max(prvals_in[prvals_in[:, 0] >= precision][:, 1])
    except (ValueError, IndexError):
        return 0.0


def getPR(mInds, gt, locRad):
    positives = np.argwhere(mInds != -1)[:, 0]
    tp = np.sum(gt[positives] <= locRad)
    fp = len(positives) - tp

    negatives = np.argwhere(mInds == -1)[:, 0]
    tn = np.sum(gt[negatives] > locRad)
    fn = len(negatives) - tn

    if tp + tn + fp + fn != len(gt):
        print(f"tp ({tp}) + tn ({tn}) + fp ({fp}) + fn ({fn}) != len(gt) ({len(gt)})")
        raise ValueError("tp + tn + fp + fn != len(gt)")

    if tp == 0:
        return 0, 0, 0  # what else?

    prec = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    fscore = 2 * prec * recall / (prec + recall)

    return prec, recall, fscore


def getPRCurve(mInds, mDists, gt, locRad):
    num_threshold = 100
    prfData = []
    for thresh in np.linspace(start=mDists.min(), stop=mDists.max(), num=num_threshold).astype(np.float32):
        matchFlags = mDists <= thresh
        outVals = mInds.copy()
        outVals[~matchFlags] = -1

        p, r, f = getPR(outVals, gt, locRad)
        prfData.append([p, r, f])
    return np.array(prfData)


def getPRCurveWrapperFromDistMatrix(dist_matrix, gt_tolerance, dist_to_gt=None):
    assert len(dist_matrix.shape) == 2
    return getPRCurveWrapperFromScores(np.argmin(dist_matrix, axis=0), np.min(dist_matrix, axis=0), gt_tolerance, dist_to_gt)


def getPRCurveWrapperFromScores(dist_indices, dist_scores, gt_tolerance, dist_to_gt=None):
    if dist_to_gt is None:
        dist_to_gt = np.abs(np.arange(len(dist_indices)) - dist_indices)
    prvals = np.array(getPRCurve(dist_indices, dist_scores, dist_to_gt, gt_tolerance))
    return prvals
