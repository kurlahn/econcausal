import hashlib
import importlib.util
from typing import Dict, List, Protocol, Tuple

from econcausal.keel import Triplet
from econcausal.slipway.corpus import cue_table


class ExtractionClient(Protocol):
    def emit(self, sentence: str, pass_index: int) -> List[Triplet]: ...


class CueGrammarExtractor:
    def __init__(self, names: Tuple[str, ...], dropout: float) -> None:
        self._index: Dict[str, int] = {name: pos for pos, name in enumerate(names)}
        self._cues: List[Tuple[str, str]] = sorted(
            cue_table().items(), key=lambda item: -len(item[0])
        )
        self._dropout = dropout

    def emit(self, sentence: str, pass_index: int) -> List[Triplet]:
        body = sentence[:-2] if sentence.endswith(" .") else sentence
        padded = f" {body} "
        for phrase, relation in self._cues:
            if f" {phrase} " not in padded:
                continue
            left, right = body.split(phrase, 1)
            left_tokens = left.split()
            right_tokens = right.split()
            if not left_tokens or not right_tokens:
                return []
            cause, effect = left_tokens[-1], right_tokens[0]
            if cause not in self._index or effect not in self._index:
                return []
            if self._suppressed(sentence, pass_index):
                return []
            return [Triplet(self._index[cause], relation, self._index[effect], 1.0)]
        return []

    def _suppressed(self, sentence: str, pass_index: int) -> bool:
        digest = hashlib.blake2b(f"{pass_index}:{sentence}".encode(), digest_size=8).digest()
        fraction = int.from_bytes(digest, "big") / float(1 << 64)
        return fraction < self._dropout


class GuardedLanguageClient:
    def __init__(self, provider: str = "openai") -> None:
        self._provider = provider

    def emit(self, sentence: str, pass_index: int) -> List[Triplet]:
        if importlib.util.find_spec(self._provider) is None:
            raise RuntimeError(
                f"language provider '{self._provider}' is not installed; "
                "use CueGrammarExtractor for offline extraction"
            )
        raise NotImplementedError(
            "live language extraction requires an authorised provider session"
        )
