import dataclasses

import numpy as np

from econcausal.blueprint.schema import BackendCfg, NotearsCfg
from econcausal.hull.construct import construct_regime_graph
from econcausal.keel import FloatArray
from econcausal.rigging.distance import graph_edit_distance


def permutation_test(
    data_first: FloatArray,
    data_second: FloatArray,
    prior_first: FloatArray,
    prior_second: FloatArray,
    names: "tuple[str, ...]",
    backend: BackendCfg,
    notears: NotearsCfg,
    observed: int,
    permutations: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    fast = dataclasses.replace(backend, method="pc")
    combined = np.vstack([data_first, data_second])
    split = data_first.shape[0]
    exceed = 0
    for _ in range(permutations):
        index = rng.permutation(combined.shape[0])
        left = combined[index[:split]]
        right = combined[index[split:]]
        graph_left = construct_regime_graph("a", names, left, prior_first, fast, notears).graph
        graph_right = construct_regime_graph("b", names, right, prior_second, fast, notears).graph
        if graph_edit_distance(graph_left, graph_right) >= observed:
            exceed += 1
    return (exceed + 1) / (permutations + 1)
