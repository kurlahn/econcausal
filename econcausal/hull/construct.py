from typing import Tuple

import numpy as np

from econcausal.blueprint.schema import BackendCfg, NotearsCfg
from econcausal.hull.backends import (
    lingam_backend,
    notears_backend,
    pc_backend,
    pcmci_backend,
)
from econcausal.hull.edge_filter import dual_threshold
from econcausal.keel import BoolArray, DirectedGraph, DiscoveryResult, FloatArray


def run_backend(
    data: FloatArray, prior: FloatArray, backend: BackendCfg, notears: NotearsCfg
) -> Tuple[FloatArray, FloatArray]:
    method = backend.method.lower()
    if method == "notears":
        return notears_backend(data, prior, notears)
    if method == "pc":
        return pc_backend(data)
    if method == "pcmci":
        return pcmci_backend(data, backend.max_lag)
    if method == "lingam":
        return lingam_backend(data)
    raise KeyError(f"unknown backend: {backend.method}")


def _pick(forward: float, prior_forward: float) -> float:
    if forward != 0.0:
        return forward
    if prior_forward > 0.0:
        return prior_forward
    return 1.0


def _orient(estimate: FloatArray, mask: BoolArray, prior: FloatArray) -> FloatArray:
    order = estimate.shape[0]
    final = np.zeros((order, order), dtype=np.float64)
    for i in range(order):
        for j in range(i + 1, order):
            if not (mask[i, j] or mask[j, i]):
                continue
            if abs(estimate[i, j]) >= abs(estimate[j, i]):
                final[i, j] = _pick(float(estimate[i, j]), float(prior[i, j]))
            else:
                final[j, i] = _pick(float(estimate[j, i]), float(prior[j, i]))
    return final


def _stats_only(estimate: FloatArray, pvalues: FloatArray, alpha: float) -> FloatArray:
    significant: BoolArray = pvalues < alpha
    return _orient(estimate, significant, np.zeros_like(estimate))


def construct_regime_graph(
    label: str,
    names: Tuple[str, ...],
    data: FloatArray,
    prior: FloatArray,
    backend: BackendCfg,
    notears: NotearsCfg,
) -> DiscoveryResult:
    estimate, pvalues = run_backend(data, prior, backend, notears)
    mask = dual_threshold(pvalues, prior, backend.alpha, backend.stringency)
    final = _orient(estimate, mask, prior)
    text_only = np.where(prior > 0.0, prior, 0.0).astype(np.float64)
    np.fill_diagonal(text_only, 0.0)
    stats_only = _stats_only(estimate, pvalues, backend.alpha)
    return DiscoveryResult(
        label=label,
        graph=DirectedGraph(names, final),
        prior=prior,
        text_only=DirectedGraph(names, text_only),
        stats_only=DirectedGraph(names, stats_only),
    )
