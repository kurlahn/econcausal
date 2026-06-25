from typing import Dict, List, Tuple

import numpy as np

from econcausal.keel import Triplet

_Key = Tuple[int, str, int]


def tally(passes: List[List[Triplet]], threshold: float, floor: float) -> List[Triplet]:
    rounds = len(passes)
    if rounds == 0:
        return []
    counts: Dict[_Key, int] = {}
    confidences: Dict[_Key, List[float]] = {}
    for triplets in passes:
        seen: set[_Key] = set()
        for item in triplets:
            key: _Key = (item.cause, item.relation, item.effect)
            if key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
            confidences.setdefault(key, []).append(item.confidence)
    kept: List[Triplet] = []
    for key, count in counts.items():
        fraction = count / rounds
        if fraction <= threshold:
            continue
        confidence = float(np.mean(confidences[key])) * fraction
        if confidence >= floor:
            kept.append(Triplet(key[0], key[1], key[2], confidence))
    kept.sort(key=lambda t: (t.cause, t.effect, t.relation))
    return kept
