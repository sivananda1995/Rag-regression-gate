# Defense guide: rag-regression-gate

How to talk about this project under questioning. Every number here is in `docs/metrics.json` and is
re-measured by `make verify`, so the answer to "how do you know" is always "run this command".

---

## The 30 second version

"It is a CI gate for retrieval quality. It scores a labelled golden set of 140 queries on every pull
request and exits non-zero on a real regression, and the interesting half is that it refuses to fire on
noise: a drop has to clear an agreed tolerance **and** a 95% paired bootstrap interval that excludes
zero. In the included demo it catches a chunking change that costs 6.8 points of recall@5 and takes 5
of 140 questions from answered to unanswered, and it reports a borderline case as WARN with an
instruction to enlarge the golden set rather than failing the build. The same machinery confirmed the
one improvement that shipped and threw away two that did not. 186 tests, 92% line coverage, every number
re-measured by CI."

Then stop. The next question is usually "how do you avoid false alarms", which is the design.

---

## The four claims, and how each is proved

### 1. Retrieval breaks quietly, and nothing in a normal CI run notices

**The claim.** A chunker change, a model swap or a normalisation tweak moves which documents come back.
Unit tests still pass, the service still returns 200, and the failure surfaces later as users not
finding answers.

**The proof.** `ragate gate` scores a candidate pipeline against a committed baseline on the golden set
and exits 1 on a confirmed regression.

**The numbers.** Replacing sentence-window chunking with fixed 240-character slices drops recall@5 from
0.9438 to 0.8762, which is the 6.8 point drop in the badge. Underneath the average: 11 queries score
zero after the change, but only 5 of those are its fault, because 7 were already scoring zero
beforehand. 1 query that had nothing correct now finds something. So the net damage is 5 minus 1 over
140, which is 2.9% of the golden set going answerless.

**If pushed: "why not just quote the 11?"** Because 7 of them were broken before the change, and
charging a change for breakage it did not cause is how a gate loses credibility. The per-query
transition table is the artefact: answered to unanswered, unanswered to answered, and no change.

### 2. A threshold alone fires on noise, so teams switch it off

**The claim.** On a golden set of 140 queries, run-to-run and configuration-to-configuration variation
is large enough that a fixed threshold produces false alarms, and a gate that cries wolf gets disabled.

**The proof.** Every verdict requires two conditions: the delta must exceed the declared tolerance, and
a 95% paired bootstrap interval over 5,000 resamples must exclude zero. Paired, because the same
queries are scored under both pipelines and the pairing removes query difficulty from the variance.

**The numbers.** The borderline candidate in the repository has a delta of **-0.0152** with an interval
of **[-0.0386, 0.0024]**. The interval contains zero, 3 queries are affected, and the verdict is
**WARN** with an instruction to enlarge the golden set. It does not fail the build. The confirmed
regression, by contrast, has an interval of [-0.1058, -0.0340], which does not contain zero.

**If pushed: "a bootstrap on 140 queries is thin."** It is, and the WARN verdict is the honest response
to exactly that. The tool cannot manufacture statistical power it does not have; what it can do is
distinguish "this is a regression" from "this set is too small to tell", and say which one it is. The
number of resamples is fixed and seeded, so the interval is reproducible.

### 3. The same machinery confirms improvements, which is what makes it trustworthy

**The claim.** A gate that can only block is a gate people route around. The interesting use is
confirming that a change helped.

**The proof.** The reranker in this repository was accepted because the gate confirmed it, and two other
plausible improvements were measured and thrown away. ADR-004 and ADR-005 record both.

**The numbers.** The reranker earns **+0.0305** recall@5 across all queries with an interval of
**[0.0071, 0.0600]** that stays above zero, and on the held-out split it takes recall from 0.9070 to
0.9535. Out of fold on the training split it is 0.9361 against 0.9395 in sample, which is the gap that
says it is not memorising.

**Rejected: hybrid retrieval.** Fusing BM25 with the dense retriever by reciprocal rank is the standard
recommendation. BM25 alone scores 0.9133; the sweep over seven dense weights and four `rrf_k` values,
selected on the train split only, chose a dense weight of **0.0**, which is BM25 by another name. Per
query, the dense retriever beat BM25 on **zero** of 140 queries, scoring 0.8636 overall. Fusion cannot
add information a component does not have.

**Rejected: chunk-level reranking.** It scored 0.8411 on the held-out split, **-0.0659** against the
document-level reranker, because a breadth coefficient of 0.6538 means the model learned to prefer
documents with several matching chunks, which is a proxy for length rather than for relevance.

**If pushed: "you kept the code for things you rejected?"** Yes, and that is deliberate. The hybrid
implementation and the sweep stay because the question becomes live again the moment a real embedding
model is plugged in, and then it is one command rather than a rewrite. What is not kept is the claim.

### 4. Determinism, because a gate that flickers is worse than no gate

**The claim.** If the same commit can produce two different verdicts, the gate is noise.

**The proof.** ADR-006. Ranking ties are broken deterministically, the bootstrap is seeded, and the
verdict is invariant to thread count and to the hash seed, which is asserted by test.

**The numbers.** Duplicate collapse takes 1,260 chunks to 518 index units, saving 58.9% of the vectors,
and it changes the result: with duplicates in the index, parity between a 3x and 12x duplicated corpus
is 0.9368 and 0.9284, and after collapsing it is 0.9993 and 0.9986. Duplicated content was quietly
crowding out the top k.

---

## Questions that are meant to be hard

**"Exact search rather than an approximate index? That is not what production runs."** ADR-001, and it
is a deliberate boundary. An approximate index adds its own recall loss, which would be
indistinguishable from the pipeline change under test. p95 latency on the largest index here is 2.8 ms
over 12,673 vectors, so exactness is affordable at this scale, and the gate's job is to measure the
pipeline rather than to imitate the serving stack. What it cannot measure is ANN recall loss, and that
is stated rather than implied.

**"The embedder is a hashing embedder, not a real model."** Correct, and it is why the dense-only number
is 0.8636 and why hybrid fusion loses here. The gate's arithmetic does not depend on which embedder
produced the ranking: swap in a real one, re-run the sweep, and the same machinery answers the same
question. The claim this repository does not make is that dense retrieval is worse than BM25 in
general.

**"Why is the baseline a file in git rather than a database?"** ADR-002. A baseline in git is reviewed in
the pull request that changes it, which means moving the goalposts is a diff somebody has to approve.
A baseline in a service is moved by whoever has credentials, silently.

**"140 queries is small."** It is, and the WARN verdict exists because of it. The recall ceiling on this
corpus is 0.9976, so there is real headroom left to measure, and the tool reports the interval width
rather than pretending to a precision it does not have.

**"What is the weakest part?"** The golden set is the whole foundation, and it was written alongside the
corpus. A golden set derived from real user queries would have a different difficulty distribution and
probably a lower ceiling. Everything downstream, including how large a drop the gate can resolve, is
bounded by that set.

**"What would you do differently with more time?"** Grow the golden set from production query logs with
human labelling, in that order, because the interval width is the binding constraint on every verdict
the gate can issue. Then add a second metric with a different failure mode, such as answer
faithfulness, so a change that trades retrieval for generation quality cannot pass unnoticed.

---

## Things to say and things not to say

**Say:** "a drop has to clear the tolerance and the interval has to exclude zero", "5 of 140 queries
went from answered to unanswered, and 7 were already broken", "the reranker shipped because the gate
confirmed it", "the interval contains zero, so the answer is enlarge the set".

**Do not say:** "it catches retrieval regressions" without the size it can resolve, "hybrid retrieval
does not work" (it does not work here, with this embedder, on this corpus), or "92% recall" as though
recall were a single number rather than recall@5 on a named split.

---

## If a live demo is asked for

```bash
ragate eval                                              # the baseline: recall@5 0.9438 over 140 queries
ragate -c configs/candidate-fixed-chunking.yaml gate     # FAIL, exit 1, 6.8 points, 5 queries blanked
ragate -c configs/candidate-borderline.yaml gate         # WARN, exit 0, interval contains zero
ragate -c configs/candidate-reranker.yaml gate           # PASS with a confirmed gain
```

`make demo` runs that sequence, and `docs/video/` holds a recording of it.
