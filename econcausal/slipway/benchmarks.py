from typing import Dict, List, Tuple

import numpy as np

from econcausal.keel import FloatArray

_ASIA_NODES: Tuple[str, ...] = (
    "asia",
    "smoke",
    "tub",
    "lung",
    "bronc",
    "either",
    "xray",
    "dysp",
)

_ASIA_EDGES: Tuple[Tuple[str, str], ...] = (
    ("asia", "tub"),
    ("smoke", "lung"),
    ("smoke", "bronc"),
    ("tub", "either"),
    ("lung", "either"),
    ("either", "xray"),
    ("either", "dysp"),
    ("bronc", "dysp"),
)

BENCHMARKS: Dict[str, Tuple[int, int]] = {
    "asia": (8, 8),
    "child": (20, 25),
    "alarm": (37, 46),
}


def _asia_world(signal_strength: float, seed: int) -> Tuple[Tuple[str, ...], FloatArray]:
    rng = np.random.default_rng(seed)
    rank = {name: pos for pos, name in enumerate(_ASIA_NODES)}
    n = len(_ASIA_NODES)
    weights = np.zeros((n, n), dtype=np.float64)
    for source, target in _ASIA_EDGES:
        magnitude = signal_strength * float(rng.uniform(0.6, 1.4))
        weights[rank[source], rank[target]] = magnitude * float(rng.choice(np.array([-1.0, 1.0])))
    return _ASIA_NODES, weights


def _generated_world(
    n: int, target_edges: int, signal_strength: float, seed: int
) -> Tuple[Tuple[str, ...], FloatArray]:
    rng = np.random.default_rng(seed)
    names = tuple(f"N{i:02d}" for i in range(n))
    candidates: List[Tuple[int, int]] = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(candidates)
    weights = np.zeros((n, n), dtype=np.float64)
    for i, j in candidates[:target_edges]:
        magnitude = signal_strength * float(rng.uniform(0.6, 1.4))
        weights[i, j] = magnitude * float(rng.choice(np.array([-1.0, 1.0])))
    return names, weights


def benchmark_world(
    name: str, signal_strength: float, seed: int
) -> Tuple[Tuple[str, ...], FloatArray]:
    key = name.lower()
    if key not in BENCHMARKS:
        raise KeyError(f"unknown benchmark: {name}")
    if key == "asia":
        return _asia_world(signal_strength, seed)
    nodes, edges = BENCHMARKS[key]
    return _generated_world(nodes, edges, signal_strength, seed)
