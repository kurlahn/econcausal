import dataclasses
from typing import Tuple


@dataclasses.dataclass(frozen=True)
class DataCfg:
    n_variables: int
    n_regimes: int
    documents_per_regime: int
    series_length: int
    edge_density: float
    signal_strength: float
    noise_scale: float
    decoy_rate: float
    seed: int


@dataclasses.dataclass(frozen=True)
class ExtractionCfg:
    passes: int
    vote_threshold: float
    confidence_floor: float
    dropout: float


@dataclasses.dataclass(frozen=True)
class ConstraintCfg:
    sign_restrictions: bool
    temporal_precedence: bool
    accounting_identities: bool


@dataclasses.dataclass(frozen=True)
class PriorCfg:
    sources: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class NotearsCfg:
    l1_penalty: float
    prior_penalty: float
    learning_rate: float
    outer_steps: int
    inner_steps: int
    rho_init: float
    rho_max: float
    h_tolerance: float
    progress_rate: float


@dataclasses.dataclass(frozen=True)
class BackendCfg:
    method: str
    alpha: float
    stringency: float
    max_lag: int
    prior_relaxation: float


@dataclasses.dataclass(frozen=True)
class ComparisonCfg:
    permutations: int
    top_paths: int
    instruments: Tuple[str, ...]
    targets: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class RunCfg:
    epochs: int
    batch_size: int
    grad_accum: int
    world_size: int
    learning_rate: float
    warmup: int
    weight_decay: float
    precision: str
    seeds: int
    base_seed: int


@dataclasses.dataclass(frozen=True)
class ExperimentCfg:
    name: str
    data: DataCfg
    extraction: ExtractionCfg
    constraints: ConstraintCfg
    prior: PriorCfg
    notears: NotearsCfg
    backend: BackendCfg
    comparison: ComparisonCfg
    run: RunCfg

    def effective_batch(self) -> int:
        return self.run.batch_size * self.run.grad_accum * self.run.world_size
