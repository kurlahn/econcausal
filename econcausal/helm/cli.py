import json
from typing import Optional, Tuple

import click
import numpy as np

from econcausal.blueprint.loader import build_experiment
from econcausal.drydock.runner import fit_pipeline
from econcausal.drydock.solver import solver_history
from econcausal.helm.export import export_onnx


@click.group()
def main() -> None:
    pass


def _resolve_seed(base: int, override: Optional[int]) -> int:
    return base if override is None else override


@main.command()
@click.option("--config", "-c", "config", required=True)
@click.option("--set", "overrides", multiple=True)
@click.option("--seed", "seed", type=int, default=None)
def fit(config: str, overrides: Tuple[str, ...], seed: Optional[int]) -> None:
    cfg = build_experiment(config, list(overrides))
    report = fit_pipeline(cfg, _resolve_seed(cfg.run.base_seed, seed))
    click.echo(
        json.dumps(
            {
                "name": report.name,
                "seed": report.seed,
                "headline_f1": round(report.headline_f1, 4),
                "interaction_ratio": round(report.interaction_ratio, 4),
                "effective_batch": cfg.effective_batch(),
            }
        )
    )


@main.command()
@click.option("--config", "-c", "config", required=True)
@click.option("--set", "overrides", multiple=True)
def appraise(config: str, overrides: Tuple[str, ...]) -> None:
    cfg = build_experiment(config, list(overrides))
    scores = np.array(
        [fit_pipeline(cfg, cfg.run.base_seed + i).headline_f1 for i in range(cfg.run.seeds)],
        dtype=np.float64,
    )
    click.echo(
        json.dumps(
            {
                "name": cfg.name,
                "seeds": cfg.run.seeds,
                "f1_mean": round(float(scores.mean()), 4),
                "f1_std": round(float(scores.std(ddof=1) if scores.size > 1 else 0.0), 4),
            }
        )
    )


@main.command()
@click.option("--config", "-c", "config", required=True)
@click.option("--set", "overrides", multiple=True)
@click.option("--seed", "seed", type=int, default=None)
def chart(config: str, overrides: Tuple[str, ...], seed: Optional[int]) -> None:
    cfg = build_experiment(config, list(overrides))
    report = fit_pipeline(cfg, _resolve_seed(cfg.run.base_seed, seed))
    comparison = report.comparison
    count = len(comparison.labels)
    rows = []
    for r in range(count):
        for s in range(r + 1, count):
            rows.append(
                {
                    "pair": [comparison.labels[r], comparison.labels[s]],
                    "ged": int(comparison.ged[r, s]),
                    "jaccard": round(float(comparison.jaccard[r, s]), 3),
                    "pvalue": round(float(comparison.pvalues[r, s]), 3),
                }
            )
    click.echo(json.dumps({"name": report.name, "pairs": rows}))


@main.command()
@click.option("--config", "-c", "config", required=True)
@click.option("--set", "overrides", multiple=True)
@click.option("--seed", "seed", type=int, default=None)
def trace(config: str, overrides: Tuple[str, ...], seed: Optional[int]) -> None:
    cfg = build_experiment(config, list(overrides))
    report = fit_pipeline(cfg, _resolve_seed(cfg.run.base_seed, seed))
    summary = {
        label: [{"path": path, "strength": round(strength, 4)} for path, strength in paths]
        for label, paths in report.comparison.paths.items()
    }
    click.echo(json.dumps({"name": report.name, "paths": summary}))


@main.command()
@click.option("--config", "-c", "config", required=True)
@click.option("--output", "-o", "output", default="artifacts/objective.onnx")
def export(config: str, output: str) -> None:
    cfg = build_experiment(config, [])
    export_onnx(output, cfg.data.n_variables)
    click.echo(json.dumps({"exported": output, "order": cfg.data.n_variables}))


@main.command()
@click.option("--config", "-c", "config", required=True)
@click.option("--seed", "seed", type=int, default=None)
def warmup(config: str, seed: Optional[int]) -> None:
    cfg = build_experiment(config, [])
    history = solver_history(cfg, _resolve_seed(cfg.run.base_seed, seed))
    click.echo(
        json.dumps({"name": cfg.name, "first": round(history[0], 5), "last": round(history[-1], 5)})
    )


if __name__ == "__main__":
    main()
