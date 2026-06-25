from typing import List, Set, Tuple

from econcausal.blueprint.schema import ConstraintCfg
from econcausal.keel import Triplet


def _mutual_pairs(triplets: List[Triplet]) -> Set[Tuple[int, int]]:
    directed = {(t.cause, t.effect) for t in triplets}
    flagged: Set[Tuple[int, int]] = set()
    for cause, effect in directed:
        if (effect, cause) in directed:
            flagged.add((cause, effect))
            flagged.add((effect, cause))
    return flagged


def apply_constraints(triplets: List[Triplet], cfg: ConstraintCfg) -> List[Triplet]:
    mutual = _mutual_pairs(triplets) if cfg.accounting_identities else set()
    kept: List[Triplet] = []
    for item in triplets:
        if cfg.temporal_precedence and item.cause == item.effect:
            continue
        if cfg.sign_restrictions and item.relation == "comoves":
            continue
        if cfg.accounting_identities and (item.cause, item.effect) in mutual:
            continue
        kept.append(item)
    return kept
