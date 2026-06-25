from typing import Dict, List, Tuple

import numpy as np

from econcausal.keel import FloatArray, Triplet


def prior_matrix(triplets: List[Triplet], n: int) -> FloatArray:
    accumulator: Dict[Tuple[int, int], List[float]] = {}
    for item in triplets:
        accumulator.setdefault((item.cause, item.effect), []).append(item.confidence)
    weights = np.zeros((n, n), dtype=np.float64)
    for (cause, effect), values in accumulator.items():
        weights[cause, effect] = float(np.mean(values))
    return weights
