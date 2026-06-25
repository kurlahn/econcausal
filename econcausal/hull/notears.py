from typing import List, Tuple

import numpy as np
import torch

from econcausal.blueprint.schema import NotearsCfg
from econcausal.keel import FloatArray


def acyclicity(matrix: torch.Tensor) -> torch.Tensor:
    order = matrix.shape[0]
    expm = torch.matrix_exp(matrix * matrix)
    return torch.trace(expm) - order


def _least_squares(data: torch.Tensor, weighted: torch.Tensor, samples: int) -> torch.Tensor:
    residual = data - data @ weighted
    return 0.5 / samples * (residual * residual).sum()


def _penalized(
    data: torch.Tensor,
    prior: torch.Tensor,
    masked: torch.Tensor,
    samples: int,
    cfg: NotearsCfg,
) -> torch.Tensor:
    fit = _least_squares(data, masked, samples)
    sparsity = cfg.l1_penalty * masked.abs().sum()
    relief = cfg.prior_penalty * (prior * masked.abs()).sum()
    return fit + sparsity - relief


def _inner_step(
    optimizer: "torch.optim.LBFGS",
    weights: torch.Tensor,
    off_diagonal: torch.Tensor,
    data: torch.Tensor,
    prior: torch.Tensor,
    samples: int,
    cfg: NotearsCfg,
    lagrange: float,
    rho: float,
) -> None:
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        masked = weights * off_diagonal
        constraint = acyclicity(masked)
        objective = (
            _penalized(data, prior, masked, samples, cfg)
            + lagrange * constraint
            + 0.5 * rho * constraint * constraint
        )
        objective.backward()
        return objective

    optimizer.step(closure)


def notears_solve(
    data: FloatArray, prior: FloatArray, cfg: NotearsCfg
) -> Tuple[FloatArray, List[float]]:
    observations = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float64))
    prior_tensor = torch.from_numpy(np.ascontiguousarray(prior, dtype=np.float64))
    samples, order = observations.shape
    off_diagonal = 1.0 - torch.eye(order, dtype=torch.float64)
    weights = torch.zeros((order, order), dtype=torch.float64, requires_grad=True)
    rho = cfg.rho_init
    lagrange = 0.0
    previous_h = float("inf")
    history: List[float] = [
        float(_penalized(observations, prior_tensor, torch.zeros_like(weights), samples, cfg))
    ]
    for _ in range(cfg.outer_steps):
        optimizer = torch.optim.LBFGS(
            [weights],
            lr=cfg.learning_rate,
            max_iter=cfg.inner_steps,
            line_search_fn="strong_wolfe",
        )
        _inner_step(
            optimizer,
            weights,
            off_diagonal,
            observations,
            prior_tensor,
            samples,
            cfg,
            lagrange,
            rho,
        )
        with torch.no_grad():
            masked = weights * off_diagonal
            h_value = float(acyclicity(masked))
            history.append(float(_penalized(observations, prior_tensor, masked, samples, cfg)))
        lagrange += rho * h_value
        if h_value > cfg.progress_rate * previous_h:
            rho = min(rho * 10.0, cfg.rho_max)
        previous_h = h_value
        if h_value <= cfg.h_tolerance:
            break
    with torch.no_grad():
        estimate = (weights * off_diagonal).detach().cpu().numpy().astype(np.float64)
    return estimate, history
