import dataclasses
from typing import Dict, List, Tuple

import numpy as np

from econcausal.blueprint.schema import BackendCfg, ComparisonCfg, NotearsCfg
from econcausal.keel import DiscoveryResult, FloatArray
from econcausal.rigging.distance import graph_edit_distance, jaccard
from econcausal.rigging.paths import transmission_paths
from econcausal.rigging.permutation import permutation_test


@dataclasses.dataclass
class RegimeComparison:
    labels: Tuple[str, ...]
    ged: FloatArray
    jaccard: FloatArray
    pvalues: FloatArray
    paths: Dict[str, List[Tuple[List[int], float]]]


def compare_regimes(
    results: List[DiscoveryResult],
    series: List[FloatArray],
    names: Tuple[str, ...],
    backend: BackendCfg,
    notears: NotearsCfg,
    comparison: ComparisonCfg,
    seed: int,
) -> RegimeComparison:
    count = len(results)
    ged = np.zeros((count, count), dtype=np.float64)
    jac = np.eye(count, dtype=np.float64)
    pvalues = np.ones((count, count), dtype=np.float64)
    for r in range(count):
        for s in range(r + 1, count):
            distance = graph_edit_distance(results[r].graph, results[s].graph)
            overlap = jaccard(results[r].graph, results[s].graph)
            pvalue = permutation_test(
                series[r],
                series[s],
                results[r].prior,
                results[s].prior,
                names,
                backend,
                notears,
                distance,
                comparison.permutations,
                seed + 13 * (r + 1) + s,
            )
            ged[r, s] = ged[s, r] = float(distance)
            jac[r, s] = jac[s, r] = overlap
            pvalues[r, s] = pvalues[s, r] = pvalue
    paths = {
        result.label: transmission_paths(
            result.graph, comparison.instruments, comparison.targets, comparison.top_paths
        )
        for result in results
    }
    return RegimeComparison(
        labels=tuple(r.label for r in results), ged=ged, jaccard=jac, pvalues=pvalues, paths=paths
    )
