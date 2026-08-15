"""Pure organism-vocabulary build/lookup for the nn.Embedding key space."""
from itertools import count


class OrganismVocabError(ValueError):
    """Raised on an empty organism list, or a blank/whitespace organism
    string."""


def build_vocab(organisms: list[str], unknown_index: int = 0) -> dict[str, int]:
    """Unique organisms (sorted for determinism) get sequential indices
    starting at 0; unknown_index (default 0) is reserved and never
    assigned to a real organism -- lets an organism added later (once
    its threshold is filled in) degrade to the OOV embedding at
    inference instead of crashing. Raises OrganismVocabError on empty
    input or a blank organism string."""
    if not organisms:
        raise OrganismVocabError("organisms list is empty")
    unique = sorted(set(organisms))
    for o in unique:
        if not o.strip():
            raise OrganismVocabError(f"blank organism string: {o!r}")
    candidate_indices = (i for i in count() if i != unknown_index)
    return {o: idx for o, idx in zip(unique, candidate_indices)}


def encode(vocab: dict[str, int], organism: str, unknown_index: int = 0) -> int:
    """vocab.get(organism, unknown_index). Never raises -- OOV is a
    normal, handled case, not an error."""
    return vocab.get(organism, unknown_index)
