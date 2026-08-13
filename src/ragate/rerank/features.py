"""Features for the learned reranker, computed per candidate document.

The unit of ranking is a document, not a chunk, and that is the whole point of this
module. The first version of this reranker scored chunks, because chunks are what
retrieval returns, and it made held-out recall@5 worse: 0.9070 down to 0.8411. The
coefficients said why. The strongest positive weight landed on the number of documents a
chunk belongs to, which is not a relevance signal at all. A chunk shared by thirty
articles has thirty chances of touching one of the labeled documents, so the label
rewarded promoting boilerplate. Meanwhile the metric being optimised, recall@5, counts
distinct documents. The model was solving a different problem from the one being scored.

So features are aggregated to the document before scoring: for each candidate document,
the evidence from every retrieved chunk that belongs to it is combined, the label becomes
"is this document relevant", and the model ranks the same objects the metric counts.

Two design constraints kept from the first version:

  * No component-retriever features. The reranker sees the fused score and the rank, not
    which retriever produced them. Otherwise it would learn a better fusion weighting,
    which belongs in the fusion step.
  * Every feature is cheap and nameable. A reranker whose coefficients cannot be read in
    a pull request is a reranker nobody will trust enough to keep.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..corpus import IndexUnit

FEATURE_NAMES = (
    "best_reciprocal_rank",
    "chunks_hit",
    "max_fused_score",
    "sum_fused_score",
    "max_token_coverage",
    "max_idf_weighted_coverage",
    "max_bigram_coverage",
    "max_lead_overlap",
    "code_token_match",
    # Sibling promotion. Queries that name only a product (86 of the 140 golden queries)
    # are answered by that product's article on every platform, and those variants are
    # near-identical text that crowds itself out just below the cut: their recall@5 is
    # 0.859 while recall@40 is 1.000. Similarity to the top-ranked document is how a
    # variant of the best hit is recognised without any metadata plumbing.
    "top1_lead_jaccard",
    "mean_chunk_length",
)

_TOKEN = re.compile(r"[a-z0-9]+")
# Support articles and logs name faults with codes like STG-8815 or MFA-4021. An exact
# code match is a strong relevance signal that plain token overlap dilutes, so it gets
# its own feature.
_CODE = re.compile(r"\b[A-Z]{2,5}-\d{2,6}\b")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bigrams(items: Sequence[str]) -> set[tuple[str, str]]:
    return set(zip(items, items[1:], strict=False))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class FeatureContext:
    """Corpus statistics the features need, computed once per evaluation run."""

    unit_tokens: list[list[str]]
    unit_bigrams: list[set[tuple[str, str]]]
    unit_lead: list[set[str]]
    unit_codes: list[set[str]]
    idf: dict[str, float]

    @classmethod
    def build(cls, units: Sequence[IndexUnit]) -> FeatureContext:
        unit_tokens = [tokens(unit.text) for unit in units]
        document_frequency: Counter[str] = Counter()
        for token_list in unit_tokens:
            document_frequency.update(set(token_list))
        total = len(units)
        idf = {
            term: math.log((1.0 + total) / (1.0 + df)) + 1.0
            for term, df in document_frequency.items()
        }
        return cls(
            unit_tokens=unit_tokens,
            unit_bigrams=[bigrams(tl) for tl in unit_tokens],
            # The first sentence of a chunk carries the article title on a document's
            # first chunk, which is where the product and platform live.
            unit_lead=[set(tokens(unit.text.split(".")[0])) for unit in units],
            unit_codes=[set(_CODE.findall(unit.text)) for unit in units],
            idf=idf,
        )


def extract_documents(
    query: str,
    ranking: Sequence[tuple[int, float]],
    units: Sequence[IndexUnit],
    ctx: FeatureContext,
) -> tuple[list[str], np.ndarray]:
    """Aggregate a chunk ranking into candidate documents and their feature matrix.

    Returns the candidate document ids in retrieval order (best chunk first) and a
    matrix of shape (documents, len(FEATURE_NAMES)) aligned to it. Retrieval order is
    preserved so that a model score of exactly zero leaves the ranking untouched, and so
    that ties are broken the same way on every run.
    """
    query_tokens = tokens(query)
    query_set = set(query_tokens)
    query_bigrams = bigrams(query_tokens)
    query_codes = set(_CODE.findall(query))
    query_idf_total = sum(ctx.idf.get(term, 1.0) for term in query_set) or 1.0

    # Per-chunk signals, computed once per retrieved chunk rather than per document,
    # since one chunk can belong to many documents.
    per_unit: dict[int, dict[str, float]] = {}
    order: list[str] = []
    seen: set[str] = set()
    grouped: dict[str, list[tuple[int, int, float]]] = defaultdict(list)

    for rank, (unit_index, fused_score) in enumerate(ranking, start=1):
        if unit_index not in per_unit:
            unit_set = set(ctx.unit_tokens[unit_index])
            matched = query_set & unit_set
            per_unit[unit_index] = {
                "token_coverage": len(matched) / max(len(query_set), 1),
                "idf_coverage": (
                    sum(ctx.idf.get(term, 1.0) for term in matched) / query_idf_total
                ),
                "bigram_coverage": (
                    len(query_bigrams & ctx.unit_bigrams[unit_index])
                    / max(len(query_bigrams), 1)
                ),
                "lead_overlap": (
                    len(query_set & ctx.unit_lead[unit_index]) / max(len(query_set), 1)
                ),
                "code_match": (
                    1.0 if (query_codes and query_codes & ctx.unit_codes[unit_index])
                    else 0.0
                ),
                "length": float(len(ctx.unit_tokens[unit_index])),
            }
        for doc_id in units[unit_index].doc_ids:
            if doc_id not in seen:
                seen.add(doc_id)
                order.append(doc_id)
            grouped[doc_id].append((rank, unit_index, fused_score))

    top1_lead = ctx.unit_lead[ranking[0][0]] if ranking else set()

    rows = np.zeros((len(order), len(FEATURE_NAMES)), dtype=np.float64)
    for row, doc_id in enumerate(order):
        hits = grouped[doc_id]
        signals = [per_unit[unit_index] for _rank, unit_index, _ in hits]
        best_rank = min(rank for rank, _, _ in hits)
        rows[row] = (
            1.0 / best_rank,
            math.log1p(len(hits)),
            max(score for _, _, score in hits),
            sum(score for _, _, score in hits),
            max(s["token_coverage"] for s in signals),
            max(s["idf_coverage"] for s in signals),
            max(s["bigram_coverage"] for s in signals),
            max(s["lead_overlap"] for s in signals),
            max(s["code_match"] for s in signals),
            max(jaccard(ctx.unit_lead[unit_index], top1_lead)
                for _rank, unit_index, _ in hits),
            math.log1p(sum(s["length"] for s in signals) / len(signals)),
        )
    return order, rows
