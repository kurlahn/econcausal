import numpy as np

from econcausal.blueprint.schema import ConstraintCfg
from econcausal.hull.edge_filter import dual_threshold
from econcausal.keel import Triplet
from econcausal.loft.constraints import apply_constraints
from econcausal.loft.voting import tally


def test_dual_threshold_supported_and_unsupported():
    pvalues = np.array([[1.0, 0.01, 0.20], [0.001, 1.0, 0.30], [0.40, 0.02, 1.0]])
    prior = np.array([[0.0, 0.8, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mask = dual_threshold(pvalues, prior, 0.05, 10.0)
    assert mask[0, 1]
    assert mask[1, 0]
    assert not mask[2, 1]
    assert not mask.diagonal().any()


def test_voting_keeps_majority():
    passes = [
        [Triplet(0, "raises", 1, 1.0), Triplet(1, "reduces", 2, 1.0)],
        [Triplet(0, "raises", 1, 1.0)],
        [Triplet(0, "raises", 1, 1.0), Triplet(3, "raises", 4, 1.0)],
    ]
    kept = tally(passes, threshold=0.5, floor=0.3)
    keys = {(t.cause, t.relation, t.effect) for t in kept}
    assert (0, "raises", 1) in keys
    assert (1, "reduces", 2) not in keys
    assert (3, "raises", 4) not in keys


def test_constraints_remove_decoys_self_loops_and_mutuals():
    triplets = [
        Triplet(0, "comoves", 1, 0.9),
        Triplet(2, "raises", 2, 0.9),
        Triplet(3, "raises", 4, 0.9),
        Triplet(4, "reduces", 3, 0.9),
        Triplet(5, "raises", 6, 0.9),
    ]
    cfg = ConstraintCfg(
        sign_restrictions=True, temporal_precedence=True, accounting_identities=True
    )
    kept = apply_constraints(triplets, cfg)
    pairs = {(t.cause, t.effect) for t in kept}
    assert (0, 1) not in pairs
    assert (2, 2) not in pairs
    assert (3, 4) not in pairs and (4, 3) not in pairs
    assert (5, 6) in pairs
