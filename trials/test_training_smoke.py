import json
import pathlib

from click.testing import CliRunner

from econcausal.drydock.runner import fit_pipeline
from econcausal.drydock.solver import solver_history
from econcausal.helm.cli import main
from trials._fixtures import tiny_experiment

_SMOKE = str(
    pathlib.Path(__file__).resolve().parents[1] / "drawings" / "experiment" / "_smoke.yaml"
)


def test_solver_loss_decreases_over_two_steps():
    cfg = tiny_experiment()
    history = solver_history(cfg, cfg.run.base_seed)
    assert len(history) >= 2
    assert history[-1] < history[0]


def test_pipeline_runs_end_to_end():
    cfg = tiny_experiment()
    report = fit_pipeline(cfg, cfg.run.base_seed)
    assert 0.0 <= report.headline_f1 <= 1.0
    assert len(report.regimes) == cfg.data.n_regimes


def test_cli_fit_emits_json():
    runner = CliRunner()
    result = runner.invoke(main, ["fit", "--config", _SMOKE])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "headline_f1" in payload
    assert payload["effective_batch"] == 4
