import dataclasses
from typing import List

from econcausal.blueprint.schema import ExperimentCfg
from econcausal.drydock.seeding import set_seed
from econcausal.frames.prior import prior_matrix
from econcausal.hull.notears import notears_solve
from econcausal.loft.extract import run_extraction
from econcausal.slipway.regimes import assemble_regimes


def solver_history(cfg: ExperimentCfg, seed: int) -> List[float]:
    set_seed(seed)
    cohort = assemble_regimes(dataclasses.replace(cfg.data, seed=seed))
    triplets = run_extraction(cohort, cfg.extraction, cfg.constraints)
    prior = prior_matrix(triplets[0], len(cohort.names))
    _, history = notears_solve(cohort.regimes[0].series, prior, cfg.notears)
    return history
