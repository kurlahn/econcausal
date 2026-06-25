import numpy as np

from econcausal.keel import BoolArray, FloatArray


def dual_threshold(
    pvalues: FloatArray, prior: FloatArray, alpha: float, stringency: float
) -> BoolArray:
    supported = (pvalues < alpha) & (prior > 0.0)
    unsupported = (pvalues < alpha / stringency) & (prior <= 0.0)
    retained: BoolArray = supported | unsupported
    np.fill_diagonal(retained, False)
    return retained
