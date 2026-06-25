from typing import Dict, List, Set, Tuple

from econcausal.keel import DirectedGraph

_MAX_DEPTH = 6


def transmission_paths(
    graph: DirectedGraph,
    instruments: Tuple[str, ...],
    targets: Tuple[str, ...],
    top_k: int,
) -> List[Tuple[List[int], float]]:
    lookup = {name: idx for idx, name in enumerate(graph.nodes)}
    sources = [lookup[name] for name in instruments if name in lookup]
    sinks: Set[int] = {lookup[name] for name in targets if name in lookup}
    weights = graph.weights
    adjacency: Dict[int, List[int]] = {
        node: [other for other in range(graph.size) if weights[node, other] != 0.0]
        for node in range(graph.size)
    }
    found: List[Tuple[List[int], float]] = []

    def walk(node: int, trail: List[int], strength: float, visited: Set[int]) -> None:
        if node in sinks and len(trail) > 1:
            found.append((list(trail), strength))
        if len(trail) >= _MAX_DEPTH:
            return
        for nxt in adjacency[node]:
            if nxt in visited:
                continue
            visited.add(nxt)
            trail.append(nxt)
            walk(nxt, trail, strength * abs(float(weights[node, nxt])), visited)
            trail.pop()
            visited.discard(nxt)

    for start in sources:
        walk(start, [start], 1.0, {start})
    found.sort(key=lambda item: -item[1])
    return found[:top_k]
