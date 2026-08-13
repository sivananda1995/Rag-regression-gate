# ADR-006: Ranking is a total order, quantised, with an explicit tie break

Status: accepted
Date: 2026-08-13

## Context

This repository's CI re-measures every number in the README on every push and fails if the
document and the measurements disagree. On the first push to GitHub it failed, and it was
right to. Three values differed between a GitHub runner and the machine the repository was
built on, with identical code, identical data, and the same committed model:

| value | development machine | GitHub runner |
| --- | --- | --- |
| candidate profile recall@5 | 0.8794 | 0.8826 |
| candidate delta | -0.0644 | -0.0612 |
| 95% bootstrap interval | [-0.1017, -0.0318] | [-0.0980, -0.0293] |

A quality gate whose verdict depends on which machine ran it is worthless, so this was a
do-not-ship defect rather than a rounding curiosity.

## What the cause was, and was not

It was not the interpreter's hash seed. Running the candidate under four different
`PYTHONHASHSEED` values produced 0.879405 every time, which ruled out the class of bug this
project already had once (see the war story in the README, where set iteration order leaked
into a hash function).

It was ties. Counting them settled it: **24 of the 140 golden queries have an exact tie at
the k=5 boundary**, because a templated corpus produces documents whose BM25 scores are equal
to the bit. The retriever then used `np.argpartition` to take the top k, and `argpartition`
does not define which of several equal elements lands inside the cut, so the result is
whatever its selection algorithm happens to do on that machine.

Worth recording what did **not** reproduce it, because it changes what the fix had to be. The
obvious hypothesis was the numpy version, so both 1.26.4 and 2.4.4 were run against the
pre-fix code here: both returned 0.879405, the local value. So the runner differs in something
else, most likely CPU feature dispatch inside the partition kernel, which selects different
SIMD paths on different hardware. That is not worth chasing further, because it is not
knowable from here and it is not the actual problem.

The actual problem is that the ranking had no defined answer for a tie, so *any* difference in
the machine was free to change it. That is provable locally without reproducing the runner:
multiplying every score by `1 + 1e-12 * noise`, the scale at which two machines' arithmetic
differs, moved recall@5 from 0.828929 to 0.830714. The ranking was balanced on knife edges,
and the fix had to remove the knife edges rather than pin the machine.

## Options considered

**A. Pin numpy exactly.** The first thing anyone reaches for, and here it would not even have
worked: both numpy versions produce the local value, so the runner's difference comes from
somewhere a pin does not reach. Even where pinning does mask a difference, it leaves the
ranking undefined, so the next bump silently changes published numbers and any consumer on
different hardware gets different verdicts.

**B. Loosen the receipts check to a numeric tolerance.** Would make CI pass. It also removes
the one property that makes the check worth having, and it would have hidden a real defect:
the tool genuinely was not reproducible.

**C. Make the ordering a total order.** Quantise scores before comparison so that last-bit
arithmetic differences cannot reorder anything, then break remaining ties on an explicit
secondary key, so equal scores always resolve the same way on every machine.

**D. Sort by score with a stable sort.** Better than argpartition, and not enough: stability
preserves input order for equal elements, but the input order into the sort is itself
produced by array operations, and float noise still separates values that ought to be equal.

## Decision

Option C, in `src/ragate/ranking.py`, applied by BM25, the exact index, and the reranker:

1. Round scores to 9 decimal places (`RANK_PRECISION`) in float64 before comparing. That is
   orders of magnitude coarser than float32 noise (about 1e-7 relative) and orders of
   magnitude finer than any score difference that carries meaning (BM25 term contributions
   here are around 0.1), so it discards nothing informative.
2. Rank with `np.lexsort` using the index as a secondary key, so ties resolve by index.

BM25 also now iterates query terms in sorted order rather than set order, because
floating-point addition is not associative and set iteration order is hash-seed dependent.
That was a second, smaller source of the same class of problem.

## Consequences

- The perturbation test now passes at 1e-12 and at 1e-9: recall@5 is 0.830714 at all three
  noise levels. `tests/test_ranking.py` asserts this, along with tie ordering, so the property
  cannot regress silently.
- Cross-version agreement is now checkable rather than hoped for: the three committed profiles
  produce identical recall to six decimal places under numpy 1.26.4 and 2.4.4. The candidate
  profile is also bit-identical across BLAS thread counts of 1, 2, 4 and 8 crossed with four
  interpreter hash seeds, which is the nearest available proxy for the runner's different CPU
  dispatch: `OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t ragate -c
  configs/candidate-fixed-chunking.yaml eval -o /tmp/$t.json`. Before the fix, a 1e-12
  perturbation was enough to move the metric; after it, nothing local moves it at all.
- Every number in the README moved slightly, because the tie ordering changed. That is
  expected and the new values are the reproducible ones. The candidate profile now scores
  0.8762 everywhere rather than 0.8794 here and 0.8826 there.
- It costs performance. `lexsort` is a full sort, so exact search went from O(n) to
  O(n log n) per query: p95 at 12,673 vectors rose from 1.39 ms to 2.836 ms. Paying 1.4 ms
  to make a verdict reproducible is not a close call, and the tail actually improved, since
  p99 is now 1.28x p50 rather than 4x.
- The quantisation is a real trade, not a free lunch: two candidates whose scores differ by
  less than 1e-9 are now declared equal and ordered by index. On this corpus nothing
  meaningful sits in that band, and if a future retriever produced scores that small, the
  right response would be to rescale the scores rather than to raise the precision.
- Trigger to revisit: a retriever whose meaningful score differences approach 1e-9, or a
  corpus large enough that a full sort per query dominates the run. At that point the answer
  is a partial sort over a candidate set that is first widened to include all tied values,
  which is more code for the same guarantee.
