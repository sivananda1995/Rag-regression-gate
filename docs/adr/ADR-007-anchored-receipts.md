# ADR-007: measured numbers are pinned to the sentence that claims them, in every file that claims one

- **Status:** accepted
- **Date:** 2026-08-13
- **Supersedes part of:** the receipts design described in ADR-002's "reviewed in the pull request" note
- **Related:** [ADR-006](ADR-006-deterministic-ranking.md), whose fix moved every number in the README and exposed this

## Context

`tools/collect_metrics.py` re-measures every value this repository publishes and writes it to
`docs/metrics.json`. `tools/check_readme_numbers.py` then verified each one like this:

```python
if not any(variant in readme for variant in record["display"]):
    missing.append(name)
```

For a value like `0.8762` that is a genuine assertion: the string is specific enough that
finding it means the README really does quote this build's measurement. The check ran in CI,
the README carried a badge saying so, and the badge was the third thing a reader saw.

Immediately after the ADR-006 fix landed, `make receipts` printed `every number in the readme
matches a value this build measured`, and three claims in that same README were false.

| Claim | Document said | Build measured |
| --- | --- | --- |
| queries named in the blame table | 10 | 11 |
| tests run by `make verify` | 159 | 170 |
| candidate profile recall@5 (config comment) | 0.8794 | 0.8762 |

The first two passed because `"10"` and `"159"` do occur in a document of this length: `"10"`
inside `0.9070`, `1.00`, and `10080`; `"159"` nowhere it mattered, but `test_count`'s display
string `"170"` appears in the badge, so the check found *a* 170 and was satisfied while the
Quickstart still said 159. The third was never examined at all: the tool's name says README,
and the value was in `configs/candidate-fixed-chunking.yaml`.

The failure mode is worth naming precisely, because it is not "the check had a bug". The check
did exactly what it was written to do. What it *reported* — every number matches — was a
stronger statement than what it verified — every measured value appears somewhere in one file.
A check whose output overstates its own coverage is worse than no check, because it stops
anyone from looking.

## Decision

Two changes, both narrow.

**1. A metric may pin its value to a phrase.** `ANCHORS` in `tools/collect_metrics.py` maps a
metric name to one or more templates with a single `{}`:

```python
"fail_regression_regressed_queries": ["{} queries named in the blame table"],
"test_count": ["badge/tests-{}-", "{} tests, 92% line coverage", "lint, {} tests, and"],
```

The checker substitutes the measured value and requires the whole phrase verbatim. Every
template listed must be found, which is how `test_count` now covers the badge, the coverage
sentence, and the Quickstart line independently — the three places it is written down, and the
reason the stale 159 survived.

**A metric with a short display string and no anchor is a failure**, not a pass. The threshold
is six characters, below which a display string is common enough in prose that locating it
proves nothing. This is the part that stops the weak form returning: adding a new count to the
registry without saying where it is claimed breaks the build with a message naming the metric.

**2. The registry lists every document that quotes a measurement.** `checked_documents`
currently holds `README.md`, `ragate.yaml`, `configs/candidate-fixed-chunking.yaml`, and
`src/ragate/rerank/features.py`. An anchor is satisfied by any of them, so a claim is checked
wherever it lives rather than only where the tool's name suggests.

Numbers that are not worth anchoring in a second place are simply removed from it. The
candidate profile's header comment used to restate the gate's verdict in full; it now describes
the *shape* of the outcome and points at `make receipts`. Deleting a duplicate is a better fix
than checking it.

## Consequences

Applying the check honestly to its own repository immediately produced a fourth finding, and it
was not a stale number. The executive summary said:

> leaves 10 of 140 labeled questions with no correct document in the top 5 … roughly 360
> searches a day that stop finding their answer

Eleven queries score zero under the candidate profile, so `10` was wrong. But `11` was not the
right number either: 7 queries score zero on the *baseline* as well, and one of those is fixed
by the candidate. The refactor moves 5 queries from an answer to none and recovers 1, a net
cost of 4 queries, 2.9% of the golden set, roughly 140 searches a day on the same 5,000-query
assumption. The published figure overstated the impact by a factor of two and a half by
charging the change for breakage that predated it.

No amount of number-checking finds that, because every number was arithmetically right and the
inference on top of them was wrong. What finds it is measuring the quantity the sentence is
actually about, so the gate now reports it: `GateVerdict.blanked_queries` and
`recovered_queries` name the queries that crossed between "some correct document" and "none" in
each direction, `net_blanked` subtracts them, and both report renderers print a blast-radius
line. The candidate's total zero count is deliberately *not* the headline; `net_blanked`'s
docstring says why.

Costs accepted:

- Anchors are prose, so rewording a sentence can break the build. This is the intended
  trade — the alternative is prose that drifts from its own numbers — but it means an editorial
  change to the README is a code change with a check attached.
- `checked_documents` must be extended when a new file starts quoting measurements, and nothing
  detects that automatically. Mitigated by preferring deletion: a measured value in a comment
  is usually better replaced by a pointer to `make receipts`.
- The check still cannot detect a *wrong number that no metric claims*. It verifies that every
  measured value appears where it should, not that every numeral in the document is measured.
  Closing that would mean parsing prose for numerals and asking which metric each belongs to,
  which produces false positives on every port number and year. The mitigation is the anchor
  requirement: any count worth publishing gets registered, and registration forces a location.

## Alternatives considered

**Template the README from the registry.** Guarantees consistency and produces a document
nobody wants to read; the numbers would be right and the argument around them would be
generated. Rejected for the same reason ADR-002 keeps the baseline in git rather than a
tracking server: reviewability beats automation here.

**Fail on any numeral in the README that no metric claims.** Would have caught all four
defects. It also flags `5,000` (an explicit assumption), `240` (a config value), `2026` (a
date), and every version number, so the signal drowns. Considered again with an allowlist and
rejected: an allowlist of numerals is a second document to keep in sync, and it rots the same
way.

**Line-level anchoring instead of phrase-level.** Require the value on the same line as a
keyword. Simpler, and too coarse for this README: the executive summary is one long line
containing `5`, `1`, `11`, `140`, `2.9%`, and `5,000`, so "the line mentioning blast radius
contains 5" is satisfied by `5,000`. Phrase templates are the smallest thing that actually
distinguishes the claims.
