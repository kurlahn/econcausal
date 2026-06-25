import dataclasses
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclasses.dataclass(frozen=True)
class Triplet:
    cause: int
    relation: str
    effect: int
    confidence: float


@dataclasses.dataclass(frozen=True)
class VariableSet:
    names: Tuple[str, ...]

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)


@dataclasses.dataclass
class DirectedGraph:
    nodes: Tuple[str, ...]
    weights: FloatArray

    @property
    def size(self) -> int:
        return len(self.nodes)

    def adjacency(self) -> BoolArray:
        mask: BoolArray = np.abs(self.weights) > 0.0
        np.fill_diagonal(mask, False)
        return mask

    def edge_set(self) -> "frozenset[Tuple[int, int]]":
        rows, cols = np.nonzero(self.adjacency())
        return frozenset(zip(rows.tolist(), cols.tolist()))


@dataclasses.dataclass
class RegimeData:
    label: str
    sentences: List[str]
    series: FloatArray
    sources: List[str]


@dataclasses.dataclass
class DiscoveryResult:
    label: str
    graph: DirectedGraph
    prior: FloatArray
    text_only: DirectedGraph
    stats_only: DirectedGraph
