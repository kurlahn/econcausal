import dataclasses
from typing import List, Tuple

import numpy as np

from econcausal.blueprint.schema import DataCfg
from econcausal.keel import FloatArray, RegimeData
from econcausal.slipway.corpus import build_world, render_documents
from econcausal.slipway.series import linear_sem_sample

_SOURCE_POOL: Tuple[str, ...] = ("FOMC", "ECB", "NBER")


@dataclasses.dataclass
class Cohort:
    names: Tuple[str, ...]
    regimes: List[RegimeData]
    truth: List[FloatArray]


def _regime_mask(base: FloatArray, drop_fraction: float, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    present = list(zip(*np.nonzero(base)))
    mask = base != 0.0
    n_drop = int(round(drop_fraction * len(present)))
    if n_drop > 0 and present:
        order = rng.permutation(len(present))
        for pick in order[:n_drop]:
            i, j = present[int(pick)]
            mask[i, j] = False
    return base * mask


def assemble_regimes(cfg: DataCfg) -> Cohort:
    names, base = build_world(cfg.n_variables, cfg.edge_density, cfg.signal_strength, cfg.seed)
    regimes: List[RegimeData] = []
    truth: List[FloatArray] = []
    for r in range(cfg.n_regimes):
        drop = 0.1 + 0.05 * r
        weights = _regime_mask(base, drop, cfg.seed + 101 * (r + 1))
        active = weights != 0.0
        series = linear_sem_sample(
            weights, cfg.series_length, cfg.noise_scale, cfg.seed + 211 * (r + 1)
        )
        sentences = render_documents(
            names,
            weights,
            active.astype(np.float64),
            cfg.documents_per_regime,
            cfg.decoy_rate,
            cfg.seed + 307 * (r + 1),
        )
        sources = [_SOURCE_POOL[k % len(_SOURCE_POOL)] for k in range(len(sentences))]
        regimes.append(
            RegimeData(label=f"regime_{r}", sentences=sentences, series=series, sources=sources)
        )
        truth.append(weights)
    return Cohort(names=names, regimes=regimes, truth=truth)
