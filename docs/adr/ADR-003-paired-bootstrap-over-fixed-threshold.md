# ADR-003: A drop must clear both a tolerance and a paired bootstrap interval

Status: accepted
Date: 2026-08-12

## Context

The first version of this gate did what most retrieval gates do: fail the build if
recall@5 drops by more than two points. On a 140-query golden set, that gate fires on
changes that are indistinguishable from noise. Two queries flipping is 1.4 points. A
gate with a false-alarm habit gets disabled, and a disabled gate is worse than none,
because the team believes it is protected.

## Options considered

**A. Fixed threshold on the aggregate.** One line of code, no statistics, and no way to
tell a real 3 point drop from a noisy one.

**B. Paired t-test on per-query scores.** Cheap and standard, but per-query recall is
bounded, heavily tied at 0.0 and 1.0, and nowhere near normal, so the test's assumptions
do not hold on exactly the data this gate sees.

**C. Paired bootstrap over per-query score differences.** Resample query ids with
replacement, recompute the mean difference thousands of times, and read the interval off
the percentiles. Assumes nothing about the distribution's shape. Costs a few milliseconds
for 5,000 resamples on 140 queries.

**D. Bootstrap alone, no tolerance.** Statistically clean, operationally wrong: with a
large enough golden set, a drop of 0.3 points becomes significant, and the team gets
blocked over a change nobody cares about.

## Decision

C and a tolerance together. A build fails only when the drop exceeds
`gate.max_absolute_drop` (an engineering decision about what matters) **and** the
bootstrap interval excludes zero (a statistical statement that the drop is real).

The pairing is what makes it sensitive: the same queries are scored in both runs, so
resampling the differences removes between-query variance, which dominates. An unpaired
comparison of two means on 140 queries would need a much larger effect to reach the same
confidence.

The two conditions produce three outcomes, and the third one is the point of the design:

| tolerance | interval excludes zero | verdict |
| --- | --- | --- |
| breached | yes | FAIL, build blocked |
| breached | no | WARN, "this golden set cannot separate that from noise; add queries" |
| not breached | either | PASS, with the reason stating whether the drop was measurable |

## Consequences

- The gate can say "I do not know", which is the honest answer for a 3 point drop on 140
  queries, and it says what to do about it: enlarge the golden set.
- The bootstrap is seeded (20260812). Re-running the gate on the same two reports gives
  the same verdict. An unseeded gate that flips between red and green on a re-run
  destroys trust faster than a wrong threshold.
- Measured example from configs/candidate-borderline.yaml: recall falls 3.0 points, past
  the 2 point tolerance, and the 95% interval is [-0.0629, +0.0007]. A threshold gate
  blocks that build. This gate warns and explains, and it is right to.
- Cost is bounded and small: 5,000 resamples over 140 paired scores is a single numpy
  operation on a 5,000 x 140 index matrix, a few milliseconds inside a run that spends
  most of its time embedding.
