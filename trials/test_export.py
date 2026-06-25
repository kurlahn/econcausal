import importlib.util
import pathlib

import pytest

from econcausal.helm.export import export_onnx


def test_export_writes_onnx(tmp_path: pathlib.Path):
    if importlib.util.find_spec("onnx") is None:
        pytest.skip("onnx not installed")
    target = tmp_path / "objective.onnx"
    export_onnx(str(target), 6)
    assert target.exists()
    assert target.stat().st_size > 0
