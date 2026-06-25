from econcausal.keel import DirectedGraph


def graph_edit_distance(first: DirectedGraph, second: DirectedGraph) -> int:
    left = first.edge_set()
    right = second.edge_set()
    reversed_pairs = {(i, j) for (i, j) in left if (j, i) in right and (i, j) not in right}
    reversals = len(reversed_pairs)
    deletions = len(left - right) - reversals
    insertions = len(right - left) - reversals
    return insertions + deletions + reversals


def jaccard(first: DirectedGraph, second: DirectedGraph) -> float:
    left = first.edge_set()
    right = second.edge_set()
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)
