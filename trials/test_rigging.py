import numpy as np

from econcausal.blueprint.schema import BackendCfg
from econcausal.keel import DirectedGraph
from econcausal.rigging.distance import graph_edit_distance, jaccard
from econcausal.rigging.paths import transmission_paths
from econcausal.rigging.permutation import permutation_test
from econcausal.slipway.corpus import build_world
from econcausal.slipway.series import linear_sem_sample
from trials._fixtures import fast_notears


def _graph(nodes, edges):
    weights = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for i, j in edges:
        weights[i, j] = 1.0
    return DirectedGraph(tuple(nodes), weights)


def test_ged_counts_reversal_as_one():
    nodes = ["a", "b", "c"]
    first = _graph(nodes, [(0, 1), (1, 2)])
    second = _graph(nodes, [(0, 1), (2, 1)])
    assert graph_edit_distance(first, second) == 1
    assert abs(jaccard(first, second) - 1.0 / 3.0) < 1e-9


def test_transmission_path_recovered_on_chain():
    nodes = ("FEDFUNDS", "DGS10", "INDPRO")
    chain = _graph(nodes, [(0, 1), (1, 2)])
    paths = transmission_paths(chain, ("FEDFUNDS",), ("INDPRO",), 5)
    assert paths
    assert paths[0][0] == [0, 1, 2]


def test_permutation_pvalue_is_one_for_identical_regimes():
    names, weights = build_world(5, 0.5, 1.2, 9)
    series = linear_sem_sample(weights, 200, 0.5, 9)
    prior = np.zeros_like(weights)
    backend = BackendCfg(method="pc", alpha=0.05, stringency=10.0, max_lag=4, prior_relaxation=1.0)
    pvalue = permutation_test(
        series, series.copy(), prior, prior, names, backend, fast_notears(), 0, 16, 1
    )
    assert pvalue == 1.0
