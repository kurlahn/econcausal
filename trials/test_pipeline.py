import pathlib

from econcausal.blueprint.loader import build_experiment
from econcausal.drydock.runner import fit_pipeline
from trials._fixtures import tiny_experiment

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_same_seed_is_deterministic():
    cfg = tiny_experiment()
    first = fit_pipeline(cfg, 5)
    second = fit_pipeline(cfg, 5)
    assert first.headline_f1 == second.headline_f1
    assert first.interaction_ratio == second.interaction_ratio


def test_every_experiment_config_loads():
    folder = _ROOT / "drawings" / "experiment"
    files = sorted(folder.glob("*.yaml"))
    assert files
    for path in files:
        cfg = build_experiment(str(path), [])
        assert cfg.data.n_variables >= 4
        assert cfg.effective_batch() >= 1


def test_overrides_change_configuration():
    main = str(_ROOT / "drawings" / "experiment" / "main.yaml")
    cfg = build_experiment(main, ["data.n_variables=9", "backend.method=pc"])
    assert cfg.data.n_variables == 9
    assert cfg.backend.method == "pc"
