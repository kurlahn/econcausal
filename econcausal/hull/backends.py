from typing import List, Tuple

import numpy as np

from econcausal.blueprint.schema import NotearsCfg
from econcausal.hull.notears import notears_solve
from econcausal.hull.pvalues import partial_correlation_pvalues
from econcausal.keel import FloatArray


def _association(data: FloatArray) -> FloatArray:
    order = data.shape[1]
    covariance = np.cov(data, rowvar=False) + 1e-6 * np.eye(order)
    precision = np.linalg.pinv(covariance)
    assoc = np.zeros((order, order), dtype=np.float64)
    for i in range(order):
        for j in range(order):
            if i == j:
                continue
            denom = np.sqrt(precision[i, i] * precision[j, j])
            if denom > 0.0:
                assoc[i, j] = -precision[i, j] / denom
    return assoc


def pc_backend(data: FloatArray) -> Tuple[FloatArray, FloatArray]:
    assoc = _association(data)
    directed: FloatArray = np.triu(assoc, 1)
    return directed, partial_correlation_pvalues(data)


def pcmci_backend(data: FloatArray, max_lag: int) -> Tuple[FloatArray, FloatArray]:
    lag = max(1, min(max_lag, data.shape[0] - 2))
    future = data[lag:]
    past = data[:-lag]
    design = np.concatenate([past, np.ones((past.shape[0], 1))], axis=1)
    solution, _, _, _ = np.linalg.lstsq(design, future, rcond=None)
    coefficients: FloatArray = solution[:-1, :].astype(np.float64)
    np.fill_diagonal(coefficients, 0.0)
    return coefficients, partial_correlation_pvalues(data)


def _causal_order(data: FloatArray) -> List[int]:
    order = data.shape[1]
    residual = data - data.mean(axis=0, keepdims=True)
    remaining = set(range(order))
    sequence: List[int] = []
    while remaining:
        best = -1
        best_score = float("inf")
        for candidate in sorted(remaining):
            score = 0.0
            base = residual[:, candidate]
            variance = float(np.dot(base, base)) + 1e-12
            for other in remaining:
                if other == candidate:
                    continue
                target = residual[:, other]
                beta = float(np.dot(base, target)) / variance
                leftover = target - beta * base
                corr = float(np.corrcoef(base, leftover)[0, 1])
                score += abs(corr)
            if score < best_score:
                best_score = score
                best = candidate
        sequence.append(best)
        remaining.discard(best)
        chosen = residual[:, best]
        variance = float(np.dot(chosen, chosen)) + 1e-12
        for other in remaining:
            beta = float(np.dot(chosen, residual[:, other])) / variance
            residual[:, other] = residual[:, other] - beta * chosen
    return sequence


def lingam_backend(data: FloatArray) -> Tuple[FloatArray, FloatArray]:
    order = data.shape[1]
    sequence = _causal_order(data)
    position = {node: idx for idx, node in enumerate(sequence)}
    centered = data - data.mean(axis=0, keepdims=True)
    coefficients = np.zeros((order, order), dtype=np.float64)
    for child in range(order):
        parents = [p for p in range(order) if position[p] < position[child]]
        if not parents:
            continue
        design = centered[:, parents]
        solution, _, _, _ = np.linalg.lstsq(design, centered[:, child], rcond=None)
        for slot, parent in enumerate(parents):
            coefficients[parent, child] = float(solution[slot])
    return coefficients, partial_correlation_pvalues(data)


def notears_backend(
    data: FloatArray, prior: FloatArray, cfg: NotearsCfg
) -> Tuple[FloatArray, FloatArray]:
    estimate, _ = notears_solve(data, prior, cfg)
    return estimate, partial_correlation_pvalues(data)
