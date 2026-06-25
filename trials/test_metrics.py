import numpy as np
from scipy import stats

from econcausal.keel import DirectedGraph
from econcausal.sea_trials.inference import bootstrap_ci, cohens_d, holm_bonferroni, paired_t
from econcausal.sea_trials.scores import edge_scores, interaction_ratio, structural_hamming


def _graph(edges, n):
    weights = np.zeros((n, n), dtype=np.float64)
    for i, j in edges:
        weights[i, j] = 1.0
    return DirectedGraph(tuple(f"v{k}" for k in range(n)), weights)


def test_edge_scores_match_hand_count():
    prediction = _graph([(0, 1), (1, 2)], 4)
    truth = _graph([(0, 1), (1, 3)], 4)
    scores = edge_scores(prediction, truth)
    assert abs(scores["precision"] - 0.5) < 1e-9
    assert abs(scores["recall"] - 0.5) < 1e-9
    assert abs(scores["f1"] - 0.5) < 1e-9
    assert structural_hamming(prediction, truth) == 2


def test_interaction_ratio_above_one_signals_synergy():
    ratio = interaction_ratio(0.60, 0.40, 0.42, 0.30)
    assert ratio > 1.0


def test_paired_t_matches_scipy():
    rng = np.random.default_rng(0)
    first = rng.normal(0.6, 0.05, size=20)
    second = rng.normal(0.5, 0.05, size=20)
    statistic, pvalue = paired_t(first, second)
    ref_t, ref_p = stats.ttest_rel(first, second)
    assert abs(statistic - float(ref_t)) < 1e-9
    assert abs(pvalue - float(ref_p)) < 1e-9
    assert cohens_d(first, second) > 0.0


def test_holm_is_monotone_and_bounded():
    adjusted = holm_bonferroni([0.001, 0.02, 0.04])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_bootstrap_ci_brackets_mean():
    values = np.full(40, 0.5) + np.random.default_rng(1).normal(0, 0.01, 40)
    lower, upper = bootstrap_ci(values, 200, 0.05, 1)
    assert lower <= float(values.mean()) <= upper
