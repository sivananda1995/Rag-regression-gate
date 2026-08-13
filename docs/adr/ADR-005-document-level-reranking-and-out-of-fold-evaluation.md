# ADR-005: The reranker ranks documents, and every claim about it is out of fold

Status: accepted
Date: 2026-08-13

## Context

After retrieval, recall@5 on this golden set was 0.9133, and the headroom was known and
specific: queries that name only a product (86 of 140) scored 0.859 at k=5 while their
gold documents were all inside the top 40. The answer exists once per platform, and the
variants crowd each other out just below the cut. Nothing needed to be *found*; something
needed to be *reordered*. That is a reranking problem.

Two decisions had to be made: what the reranker ranks, and how any claim about it is
allowed to be measured.

## The first attempt, and why it failed

The obvious unit of reranking is what retrieval returns, which is chunks. Features per
chunk, a label of 1 when the chunk belongs to any relevant document, a logistic
regression, done. Measured: held-out recall@5 fell from 0.9070 to 0.8411, a loss of
0.0659, while in-sample training numbers looked fine.

The coefficients explained it. The largest positive weight, +0.6553, landed on
`unit_breadth`, the number of documents a chunk belongs to. Under a chunk-level label that
is rational: a chunk shared by thirty articles has thirty chances of intersecting the
relevant set, so the label rewards it. But recall@5 counts distinct documents, and
promoting shared boilerplate pushes the specific answer out of the top five. The model was
optimising a different objective from the one being scored, and it succeeded at it.

The failed variant is preserved as `experiments/chunk_level_reranker.py` so the story is
reproducible rather than anecdotal.

## Options considered

**A. Keep chunk-level ranking and patch the features.** Drop `unit_breadth`, add a
penalty. This treats a symptom: the label still describes chunks while the metric counts
documents, and the next feature with a spurious correlation causes the same failure.

**B. Rank documents.** Aggregate the evidence from every retrieved chunk up to the
document (best rank, chunk count, max and sum of retrieval score, max text-overlap
signals), label documents, rank documents.

**C. Pairwise or listwise learning to rank.** Better aligned with a ranking metric than
pointwise classification, and a gradient-boosted implementation would probably score
higher still. Cost: the model stops being reviewable, and this repository commits its model
to git specifically so a reviewer can see a coefficient change and ask about it.

## Decision

Option B, with a linear model, and a hard rule about evaluation.

The unit of ranking matches the unit of measurement. Eleven named features per candidate
document, coefficients stored as JSON, inference in numpy so the runtime dependency set
does not grow.

The evaluation rule: anything fitted by looking at a query may not be used to make a claim
about that query.

- Coefficients are fitted only on the 97-query train split from `data/splits.json`.
- Generalisation inside train is estimated with GroupKFold grouped by query id, so a
  query's candidate documents never straddle a fold boundary. Without the grouping, a fold
  would contain other candidates of the same query and the estimate would be optimistic.
- The shipped model is refit on all train candidates after the folds have done their job of
  estimating error, which is standard practice.
- Every gain in the README is quoted on the 43 held-out queries.

## Consequences

- Measured outcome: 4066 candidate documents from 97 train queries, 0.9395 in sample and
  0.9395 out of fold, and on the held-out split recall@5 moves from 0.9070 to 0.9535.
  In-sample matching out-of-fold is the useful signal here: eleven features and a linear
  model leave nothing to memorise.
- The gate confirms the gain rather than asserting it. Against the committed pre-reranker
  baseline the improvement is +0.0305 across all queries with a 95% interval of
  [0.0071, 0.0600], which excludes zero.
- A model that scores everything equally is a no-op, not a shuffle, because ties fall back
  to retrieval order. Enabling reranking therefore cannot be worse than a tie by accident.
- The reranker can only reorder what retrieval found. `rerank.depth` is 40 because that is
  where the gold documents for the hard query class actually sit; a depth of 10 loses most
  of the gain, and the repository has a candidate profile that shows it.
- Cost: reranking adds roughly a third to the wall time of a full evaluation, which is
  irrelevant for a gate run measured in hundreds of milliseconds and would matter in a
  serving path. A serving deployment should measure it separately.
- Feature-set changes are breaking changes. The loader refuses a model whose feature names
  differ from the code's, naming the retraining command, because silently scoring the wrong
  columns would be a hard bug to see.
