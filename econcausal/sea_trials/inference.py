from typing import List, Tuple

import numpy as np
from scipy import stats

from econcausal.keel import FloatArray


def paired_t(first: FloatArray, second: FloatArray) -> Tuple[float, float]:
    statistic, pvalue = stats.ttest_rel(first, second)
    return float(statistic), float(pvalue)


def cohens_d(first: FloatArray, second: FloatArray) -> float:
    difference = first - second
    spread = float(np.std(difference, ddof=1))
    if spread == 0.0:
        return 0.0
    return float(np.mean(difference)) / spread


def holm_bonferroni(pvalues: List[float]) -> List[float]:
    total = len(pvalues)
    order = sorted(range(total), key=lambda i: pvalues[i])
    adjusted = [0.0] * total
    running = 0.0
    for rank, idx in enumerate(order):
        scaled = (total - rank) * pvalues[idx]
        running = max(running, min(1.0, scaled))
        adjusted[idx] = running
    return adjusted


def bootstrap_ci(
    values: FloatArray, replicates: int, alpha: float, seed: int
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    size = values.shape[0]
    means = np.empty(replicates, dtype=np.float64)
    for r in range(replicates):
        sample = values[rng.integers(0, size, size=size)]
        means[r] = float(np.mean(sample))
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lower, upper
