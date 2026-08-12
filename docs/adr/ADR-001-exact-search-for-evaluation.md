# ADR-001: Exact search is the default index for evaluation, with HNSW available

Status: accepted
Date: 2026-08-12

## Context

This tool exists to answer one question in a pull request: did the retrieval pipeline
get worse? The measured drop has to be attributable to the change under review, which
means every other source of variation has to be held still.

An approximate nearest neighbour index is a source of variation. It returns almost the
right neighbours, and how close "almost" is depends on the graph, the build parameters,
and the insertion order of the vectors. If the index is approximate, a recall drop of
two points could be the new chunker, or it could be the index landing differently after
a rebuild. Those two causes lead to opposite actions, and no amount of reporting
resolves them after the fact.

## Options considered

**A. Always use the approximate index (FAISS HNSW).** Matches whatever production runs,
so the number is closer to what users experience. Cost: the gate's own noise floor now
includes ANN error, which is exactly the noise the gate cannot separate from a real
regression.

**B. Always use exact search.** Removes ANN error entirely. Cost: cost. Exact cosine
search is O(n) per query, so it grows with the corpus. Measured here at k=20 chunks,
single query at a time, 512-dim vectors, 2 vCPU: p95 of 0.15 ms at 518 vectors, 0.22 ms
at 1.6k, 0.57 ms at 6.4k, 1.39 ms at 12.7k (benchmark/results/retrieval_scaling.json).

**C. Exact by default for the gate, approximate as a selectable backend.** Two runs of
the same golden set, one per backend, and the difference is the ANN penalty in isolation.

## Decision

Option C. `index.backend=flat` is the default. The gate measures the pipeline, not the
index. `index.backend=faiss` exists so the ANN cost can be measured deliberately rather
than absorbed silently.

The scale argument does not bite at golden-set size. A golden set is hundreds to low
thousands of labeled queries against a corpus small enough to keep labels correct; at
12.7k vectors exact search costs 1.39 ms at p95, so a 140-query gate run spends under a
fifth of a second in search. Paying milliseconds to remove a confound is the right trade.

## Consequences

- The gate's verdict is reproducible: the same two reports always produce the same
  verdict, because neither the index nor the bootstrap has an unseeded random component.
- Exact search stops being viable somewhere past 10^6 vectors, where per-query cost
  reaches tens of milliseconds and a full golden-set run becomes minutes. The trigger to
  switch the gate itself to an approximate index is a golden-set run exceeding 60 seconds
  in CI; at that point the honest move is to measure the ANN penalty first with the
  parity check, then subtract it explicitly.
- Because both backends are exercised, this repository can report what the approximate
  index actually costs: retrieved-score parity of 0.9999 at 1.6k vectors and 0.9987 at
  6.4k, meaning HNSW returns 99.9% of the similarity mass that exact search would.
- One caveat, learned the hard way: those parity numbers only hold after byte-identical
  chunks are collapsed. With duplicates in the index, parity fell to 0.94 and 0.91. See
  ADR-003.
