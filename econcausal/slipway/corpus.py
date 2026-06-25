from typing import Dict, List, Set, Tuple

import numpy as np

from econcausal.keel import FloatArray

MACRO_VARIABLES: Tuple[str, ...] = (
    "FEDFUNDS",
    "DGS10",
    "MORTGAGE30US",
    "HOUST",
    "BUSLOANS",
    "INDPRO",
    "DTWEXBGS",
    "BOPGSTB",
    "SP500",
    "PCE",
    "MICH",
    "CPIAUCSL",
    "UNRATE",
    "GDPC1",
    "PAYEMS",
    "AHETPI",
    "M2SL",
    "CSUSHPISA",
    "BAMLH0A0HYM2",
    "T5YIE",
)

MACRO_EDGES: Tuple[Tuple[str, str, str], ...] = (
    ("FEDFUNDS", "DGS10", "positive"),
    ("DGS10", "MORTGAGE30US", "positive"),
    ("MORTGAGE30US", "HOUST", "negative"),
    ("FEDFUNDS", "BUSLOANS", "negative"),
    ("BUSLOANS", "INDPRO", "positive"),
    ("FEDFUNDS", "DTWEXBGS", "positive"),
    ("DTWEXBGS", "BOPGSTB", "negative"),
    ("FEDFUNDS", "SP500", "negative"),
    ("SP500", "PCE", "positive"),
    ("FEDFUNDS", "MICH", "negative"),
    ("MICH", "CPIAUCSL", "positive"),
    ("UNRATE", "CPIAUCSL", "negative"),
    ("GDPC1", "UNRATE", "negative"),
    ("GDPC1", "PAYEMS", "positive"),
    ("PAYEMS", "AHETPI", "positive"),
    ("AHETPI", "CPIAUCSL", "positive"),
    ("M2SL", "CPIAUCSL", "positive"),
    ("CSUSHPISA", "PCE", "positive"),
    ("BAMLH0A0HYM2", "BUSLOANS", "negative"),
    ("T5YIE", "DGS10", "positive"),
    ("FEDFUNDS", "M2SL", "negative"),
    ("INDPRO", "GDPC1", "positive"),
    ("BOPGSTB", "GDPC1", "positive"),
    ("HOUST", "GDPC1", "positive"),
    ("PCE", "GDPC1", "positive"),
    ("DGS10", "BUSLOANS", "negative"),
)

_POSITIVE_CUES: Tuple[str, ...] = ("raises", "increases", "lifts", "boosts", "drives up")
_NEGATIVE_CUES: Tuple[str, ...] = ("reduces", "lowers", "dampens", "cools", "drives down")
_DECOY_CUES: Tuple[str, ...] = ("moves with", "co-moves with", "tracks", "is correlated with")


def _acyclic_order(names: Tuple[str, ...], edges: Tuple[Tuple[str, str, str], ...]) -> List[str]:
    index = {name: pos for pos, name in enumerate(names)}
    adjacency: Dict[str, Set[str]] = {name: set() for name in names}
    indegree: Dict[str, int] = {name: 0 for name in names}
    for source, target, _ in edges:
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1
    order: List[str] = []
    frontier = sorted((n for n in names if indegree[n] == 0), key=lambda n: index[n])
    while frontier:
        node = frontier.pop(0)
        order.append(node)
        for child in sorted(adjacency[node], key=lambda n: index[n]):
            indegree[child] -= 1
            if indegree[child] == 0:
                frontier.append(child)
        frontier.sort(key=lambda n: index[n])
    for name in names:
        if name not in order:
            order.append(name)
    return order


def macro_world(signal_strength: float, seed: int) -> Tuple[Tuple[str, ...], FloatArray, List[str]]:
    rng = np.random.default_rng(seed)
    order = _acyclic_order(MACRO_VARIABLES, MACRO_EDGES)
    rank = {name: pos for pos, name in enumerate(order)}
    names = tuple(order)
    n = len(names)
    weights = np.zeros((n, n), dtype=np.float64)
    signs: List[str] = []
    for source, target, sign in MACRO_EDGES:
        if rank[source] >= rank[target]:
            continue
        magnitude = signal_strength * float(rng.uniform(0.5, 1.5))
        value = magnitude if sign == "positive" else -magnitude
        weights[rank[source], rank[target]] = value
        signs.append(sign)
    return names, weights, signs


def build_world(
    n_variables: int, edge_density: float, signal_strength: float, seed: int
) -> Tuple[Tuple[str, ...], FloatArray]:
    rng = np.random.default_rng(seed)
    if n_variables <= len(MACRO_VARIABLES):
        base = list(MACRO_VARIABLES[:n_variables])
    else:
        base = list(MACRO_VARIABLES) + [f"VAR{i}" for i in range(len(MACRO_VARIABLES), n_variables)]
    permutation = rng.permutation(n_variables)
    names = tuple(base[i] for i in permutation)
    weights = np.zeros((n_variables, n_variables), dtype=np.float64)
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            if rng.random() < edge_density:
                magnitude = signal_strength * float(rng.uniform(0.5, 1.5))
                weights[i, j] = magnitude * float(rng.choice(np.array([-1.0, 1.0])))
    return names, weights


def render_documents(
    names: Tuple[str, ...],
    weights: FloatArray,
    active: FloatArray,
    documents: int,
    decoy_rate: float,
    seed: int,
) -> List[str]:
    rng = np.random.default_rng(seed)
    n = len(names)
    present = [(i, j) for i in range(n) for j in range(n) if active[i, j]]
    sentences: List[str] = []
    for _ in range(documents):
        rng.shuffle(present)
        take = present[: max(1, len(present) // 2)]
        for i, j in take:
            cues = _POSITIVE_CUES if weights[i, j] >= 0 else _NEGATIVE_CUES
            cue = str(rng.choice(np.array(cues, dtype=object)))
            sentences.append(f"{names[i]} {cue} {names[j]} .")
        n_decoys = int(round(decoy_rate * len(take)))
        for _ in range(n_decoys):
            a, b = int(rng.integers(n)), int(rng.integers(n))
            if a == b or active[a, b] or active[b, a]:
                continue
            cue = str(rng.choice(np.array(_DECOY_CUES, dtype=object)))
            sentences.append(f"{names[a]} {cue} {names[b]} .")
    return sentences


def cue_table() -> Dict[str, str]:
    table: Dict[str, str] = {}
    for cue in _POSITIVE_CUES:
        table[cue] = "raises"
    for cue in _NEGATIVE_CUES:
        table[cue] = "reduces"
    for cue in _DECOY_CUES:
        table[cue] = "comoves"
    return table
