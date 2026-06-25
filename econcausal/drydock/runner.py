import dataclasses
from typing import Dict, List

import numpy as np

from econcausal.blueprint.schema import ExperimentCfg
from econcausal.drydock.logbook import get_logger
from econcausal.drydock.seeding import set_seed
from econcausal.frames.prior import prior_matrix
from econcausal.hull.construct import construct_regime_graph
from econcausal.keel import DirectedGraph, DiscoveryResult, FloatArray
from econcausal.loft.extract import run_extraction
from econcausal.rigging.compare import RegimeComparison, compare_regimes
from econcausal.sea_trials.scores import edge_scores, interaction_ratio
from econcausal.slipway.regimes import assemble_regimes


@dataclasses.dataclass
class RegimeOutcome:
    label: str
    scores: Dict[str, float]
    interaction: float


@dataclasses.dataclass
class PipelineReport:
    name: str
    seed: int
    regimes: List[RegimeOutcome]
    comparison: RegimeComparison
    headline_f1: float
    interaction_ratio: float


def fit_pipeline(cfg: ExperimentCfg, seed: int) -> PipelineReport:
    logger = get_logger("runner")
    set_seed(seed)
    cohort = assemble_regimes(dataclasses.replace(cfg.data, seed=seed))
    triplets = run_extraction(cohort, cfg.extraction, cfg.constraints)
    results: List[DiscoveryResult] = []
    series: List[FloatArray] = []
    outcomes: List[RegimeOutcome] = []
    for index, regime in enumerate(cohort.regimes):
        prior = prior_matrix(triplets[index], len(cohort.names))
        result = construct_regime_graph(
            regime.label, cohort.names, regime.series, prior, cfg.backend, cfg.notears
        )
        truth = DirectedGraph(cohort.names, cohort.truth[index])
        full = edge_scores(result.graph, truth)
        text = edge_scores(result.text_only, truth)
        stats = edge_scores(result.stats_only, truth)
        ratio = interaction_ratio(full["f1"], text["f1"], stats["f1"], 0.0)
        outcomes.append(RegimeOutcome(regime.label, full, ratio))
        results.append(result)
        series.append(regime.series)
        logger.info("regime %s f1=%.3f shd=%.0f", regime.label, full["f1"], full["shd"])
    comparison = compare_regimes(
        results, series, cohort.names, cfg.backend, cfg.notears, cfg.comparison, seed
    )
    headline = float(np.mean([o.scores["f1"] for o in outcomes]))
    overall_ratio = float(np.mean([o.interaction for o in outcomes]))
    return PipelineReport(
        name=cfg.name,
        seed=seed,
        regimes=outcomes,
        comparison=comparison,
        headline_f1=headline,
        interaction_ratio=overall_ratio,
    )
