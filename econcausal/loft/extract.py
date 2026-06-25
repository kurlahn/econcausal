from typing import List

from econcausal.blueprint.schema import ConstraintCfg, ExtractionCfg
from econcausal.keel import Triplet
from econcausal.loft.client import CueGrammarExtractor
from econcausal.loft.constraints import apply_constraints
from econcausal.loft.voting import tally
from econcausal.slipway.regimes import Cohort


def extract_regime(
    extractor: CueGrammarExtractor,
    sentences: List[str],
    extraction: ExtractionCfg,
    constraints: ConstraintCfg,
) -> List[Triplet]:
    passes: List[List[Triplet]] = []
    for index in range(extraction.passes):
        emitted: List[Triplet] = []
        for sentence in sentences:
            emitted.extend(extractor.emit(sentence, index))
        passes.append(emitted)
    voted = tally(passes, extraction.vote_threshold, extraction.confidence_floor)
    return apply_constraints(voted, constraints)


def run_extraction(
    cohort: Cohort, extraction: ExtractionCfg, constraints: ConstraintCfg
) -> List[List[Triplet]]:
    extractor = CueGrammarExtractor(cohort.names, extraction.dropout)
    return [
        extract_regime(extractor, regime.sentences, extraction, constraints)
        for regime in cohort.regimes
    ]
