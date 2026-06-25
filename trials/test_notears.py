import numpy as np
import torch

from econcausal.hull.notears import acyclicity, notears_solve
from econcausal.keel import DirectedGraph
from econcausal.sea_trials.scores import edge_scores
from econcausal.slipway.corpus import build_world
from econcausal.slipway.series import linear_sem_sample
from trials._fixtures import fast_notears


def test_acyclicity_zero_for_dag_positive_for_cycle():
    dag = torch.triu(torch.ones(5, 5, dtype=torch.float64), 1) * 0.3
    assert abs(float(acyclicity(dag))) < 1e-6
    cycle = torch.tensor([[0.0, 0.9], [0.9, 0.0]], dtype=torch.float64)
    assert float(acyclicity(cycle)) > 0.1


def test_notears_recovers_planted_dag():
    names, weights = build_world(6, 0.45, 1.4, 21)
    data = linear_sem_sample(weights, 1500, 0.4, 21)
    prior = np.zeros_like(weights)
    estimate, history = notears_solve(data, prior, fast_notears())
    predicted = DirectedGraph(names, (np.abs(estimate) > 0.15).astype(np.float64))
    truth = DirectedGraph(names, weights)
    scores = edge_scores(predicted, truth)
    assert history[-1] < history[0]
    assert scores["f1"] >= 0.7


def test_notears_gradient_reaches_weights():
    names, weights = build_world(5, 0.5, 1.2, 4)
    data = torch.from_numpy(linear_sem_sample(weights, 200, 0.5, 4))
    estimate = torch.zeros((5, 5), dtype=torch.float64, requires_grad=True)
    off = 1.0 - torch.eye(5, dtype=torch.float64)
    masked = estimate * off
    residual = data - data @ masked
    objective = 0.5 / 200 * (residual * residual).sum() + acyclicity(masked) ** 2
    objective.backward()
    assert estimate.grad is not None
    assert float(estimate.grad.abs().sum()) > 0.0
