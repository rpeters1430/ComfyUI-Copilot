"""Keyword ranking shared by the node, model and workflow searches.

Queries come from LLM tool calls and from people, so matching is forgiving:
case-insensitive, split on separators ("KSampler (Advanced)", "sdxl_vae.safetensors",
"LoraLoader|pysssss" all tokenize), and partial words count.
"""

import re
from typing import Any, Callable, Iterable, Optional, TypeVar

T = TypeVar("T")

# Words that match almost everything and would drown out the useful hits
STOPWORDS = {"a", "an", "the", "and", "or", "of", "for", "to", "in", "with", "node", "nodes",
             "workflow", "workflows", "image", "images", "图像", "图片"}

# Larger than any single field weight, so query coverage dominates the score
COVERAGE_BONUS = 100.0

_SPLIT = re.compile(r"[^0-9a-z一-鿿]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: Any) -> list[str]:
    """Lowercase words of `text`; CamelCase is split too, so "KSamplerAdvanced" finds "advanced"."""
    text = _CAMEL.sub(" ", str(text or ""))
    return [token for token in _SPLIT.split(text.lower()) if token]


def query_tokens(*parts: Any) -> list[str]:
    tokens: list[str] = []
    for part in parts:
        for item in (part if isinstance(part, (list, tuple, set)) else [part]):
            tokens.extend(token for token in tokenize(item) if token not in STOPWORDS)
    return list(dict.fromkeys(tokens))


def score(tokens: list[str], fields: Iterable[tuple[Any, float]]) -> float:
    """Relevance of an item whose searchable text is `fields` ((text, weight) pairs).

    Each query token adds the weight of the best field it appears in, doubled for a
    whole-word hit. Matching more of the query always ranks first, so "flux lora" puts
    Flux LoRAs above every other LoRA, while a query with one unknown word still finds things.
    """
    prepared = [(" ".join(tokenize(text)), weight) for text, weight in fields]
    total = 0.0
    for token in tokens:
        best = 0.0
        for text, weight in prepared:
            if text and token in text:
                best = max(best, weight * (2 if token in text.split() else 1))
        if best:
            total += COVERAGE_BONUS + best
    return total


def rank(items: Iterable[T], tokens: list[str], fields: Callable[[T], Iterable[tuple[Any, float]]],
         limit: Optional[int] = None) -> list[tuple[float, T]]:
    """Items with a positive score, best first. Without tokens every item matches equally."""
    scored = [(score(tokens, fields(item)) if tokens else 1.0, item) for item in items]
    ranked = sorted((pair for pair in scored if pair[0] > 0), key=lambda pair: -pair[0])
    return ranked[:limit] if limit else ranked
