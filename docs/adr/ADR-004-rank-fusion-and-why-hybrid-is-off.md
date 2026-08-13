# ADR-004: Fusion is by reciprocal rank, and hybrid retrieval is off by default

Status: accepted
Date: 2026-08-13

## Context

Two retrievers are available: BM25 over the chunk text, and a dense retriever built from
the reproducible hashing embedder plus a vector index. The standard advice is to run both
and fuse the results, on the reasoning that lexical and vector retrieval fail in different
ways. That advice is worth testing rather than adopting, and this repository exists to
test exactly this kind of claim.

Two questions, in order. How should two rankings be combined? And does combining them help
here?

## Options considered for the combination

**A. Normalise scores and add them.** BM25 produces an unbounded sum of idf-weighted term
contributions; cosine similarity lives in [-1, 1]. To add them, both have to be mapped
onto a shared scale, and the usual choices (min-max within a query, or z-scores) make the
mapping depend on how many strong candidates that particular query happened to have. The
weighting then varies per query in a way nobody intends and nobody can tune.

**B. Reciprocal rank fusion.** Discard magnitude, keep order:

    RRF(d) = sum over retrievers r of weight(r) / (rrf_k + rank_r(d))

A document both retrievers place near the top wins, and neither scale matters. `rrf_k`
damps the advantage of the very top ranks; 60 is the value from the original paper and is
exposed as configuration rather than buried.

**C. Learn the combination.** A model over both scores plus features. This is a reranker,
not a fusion rule, and it belongs at the reranking stage (ADR-005), where it can see the
text rather than only two numbers.

## Decision on the combination

Option B. Fusion is by rank, weights are configurable, `rrf_k` is configurable, ties break
on unit index so the ordering is deterministic across runs, and each component retrieves
`k * candidate_multiplier` results so a document ranked deep by one retriever and highly
by the other can still be fused into the top k.

## Decision on whether to use it

**Hybrid retrieval is off by default. `retriever.mode` is `bm25`.**

The sweep in `tools/tune_retrieval.py` scores seven dense weights against four `rrf_k`
values on the train split and reports the winner on the held-out split. The selected dense
weight is 0.0, which is BM25 with extra steps. Component scores across all 140 queries:
BM25 0.9133, dense 0.8636.

The per-query check is what settles it: the dense retriever beats BM25 on **zero** of the
140 golden queries. There is no subset of queries where it contributes, so fusion has
nothing to fuse. That is not a surprising result for a hashing embedder, which is a lexical
representation with collisions; it is a weaker BM25, not a different view of the problem.

## Consequences

- The default pipeline is simpler and faster than the one originally planned, which is a
  good outcome that only appeared because the alternative was measured.
- BM25, the fusion implementation, and the sweep all stay in the tree. They are not dead
  code: the moment a semantic embedding model is available behind the `Embedder` protocol,
  the same sweep answers the same question in one command, and the expected answer changes,
  because a real embedding model does fail differently from BM25.
- The negative result is documented in the README rather than quietly deleted. A portfolio
  of only successful experiments is a portfolio that was not measuring anything.
- Trigger to revisit: any change to the embedding provider, or a corpus where queries and
  documents share little surface vocabulary. On this corpus the queries name products,
  platforms, and error codes, which is exactly the situation lexical retrieval owns.
- Risk accepted: `bm25` as the default means the repository's headline number depends on a
  lexical retriever, and a reader might assume the harness cannot evaluate dense retrieval.
  The README says otherwise in the same section as the numbers.
