import numpy as np
from scipy import stats

from econcausal.keel import FloatArray


def partial_correlation_pvalues(data: FloatArray) -> FloatArray:
    samples, order = data.shape
    covariance = np.cov(data, rowvar=False)
    covariance = covariance + 1e-6 * np.eye(order)
    precision = np.linalg.pinv(covariance)
    pvalues = np.ones((order, order), dtype=np.float64)
    dof = samples - order
    if dof <= 1:
        return pvalues
    for i in range(order):
        for j in range(order):
            if i == j:
                continue
            denom = np.sqrt(precision[i, i] * precision[j, j])
            if denom <= 0.0:
                continue
            pcorr = float(np.clip(-precision[i, j] / denom, -0.9999, 0.9999))
            statistic = pcorr * np.sqrt(dof / (1.0 - pcorr * pcorr))
            pvalues[i, j] = float(2.0 * stats.t.sf(abs(statistic), dof))
    return pvalues
