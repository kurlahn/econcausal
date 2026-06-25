from econcausal.blueprint.schema import (
    BackendCfg,
    ComparisonCfg,
    ConstraintCfg,
    DataCfg,
    ExperimentCfg,
    ExtractionCfg,
    NotearsCfg,
    PriorCfg,
    RunCfg,
)


def fast_notears() -> NotearsCfg:
    return NotearsCfg(
        l1_penalty=0.05,
        prior_penalty=0.5,
        learning_rate=1.0,
        outer_steps=20,
        inner_steps=100,
        rho_init=1.0,
        rho_max=1.0e12,
        h_tolerance=1.0e-8,
        progress_rate=0.25,
    )


def tiny_experiment() -> ExperimentCfg:
    return ExperimentCfg(
        name="tiny",
        data=DataCfg(
            n_variables=6,
            n_regimes=2,
            documents_per_regime=10,
            series_length=300,
            edge_density=0.4,
            signal_strength=1.2,
            noise_scale=0.5,
            decoy_rate=0.35,
            seed=3,
        ),
        extraction=ExtractionCfg(passes=5, vote_threshold=0.4, confidence_floor=0.6, dropout=0.15),
        constraints=ConstraintCfg(
            sign_restrictions=True, temporal_precedence=True, accounting_identities=True
        ),
        prior=PriorCfg(sources=("FOMC", "ECB", "NBER")),
        notears=fast_notears(),
        backend=BackendCfg(
            method="notears", alpha=0.05, stringency=10.0, max_lag=4, prior_relaxation=1.0
        ),
        comparison=ComparisonCfg(
            permutations=12,
            top_paths=3,
            instruments=("FEDFUNDS",),
            targets=("MORTGAGE30US", "INDPRO"),
        ),
        run=RunCfg(
            epochs=4,
            batch_size=4,
            grad_accum=1,
            world_size=1,
            learning_rate=0.01,
            warmup=0,
            weight_decay=0.0,
            precision="fp32",
            seeds=2,
            base_seed=3,
        ),
    )
