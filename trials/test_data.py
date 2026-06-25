import numpy as np

from econcausal.slipway.benchmarks import benchmark_world
from econcausal.slipway.corpus import build_world
from econcausal.slipway.regimes import assemble_regimes
from econcausal.slipway.series import linear_sem_sample
from trials._fixtures import tiny_experiment


def test_world_is_acyclic_upper_triangular():
    names, weights = build_world(8, 0.4, 1.0, 11)
    assert len(names) == 8
    assert np.allclose(np.tril(weights), 0.0)


def test_series_shape_and_finite():
    _, weights = build_world(6, 0.5, 1.0, 5)
    series = linear_sem_sample(weights, 256, 0.5, 5)
    assert series.shape == (256, 6)
    assert np.isfinite(series).all()


def test_benchmark_sizes():
    names, weights = benchmark_world("asia", 1.0, 2)
    assert len(names) == 8
    assert np.count_nonzero(weights) == 8
    child_names, _ = benchmark_world("child", 1.0, 2)
    assert len(child_names) == 20


def test_regimes_carry_text_and_series():
    cohort = assemble_regimes(tiny_experiment().data)
    assert len(cohort.regimes) == 2
    for regime, truth in zip(cohort.regimes, cohort.truth):
        assert regime.series.shape[1] == len(cohort.names)
        assert len(regime.sentences) > 0
        assert np.allclose(np.tril(truth), 0.0)
