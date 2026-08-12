# ADR-002: The baseline is a JSON file in git, not a row in a tracking server

Status: accepted
Date: 2026-08-12

## Context

The gate compares a candidate run against a reference. That reference has to live
somewhere. The obvious place, given that this stack already runs MLflow for model
experiments, is a tracking server: register each evaluation as a run, tag the blessed
one, and have CI query it.

## Options considered

**A. MLflow tracking server.** Central history, a UI, comparison across months, and it
is already deployed for training runs. Costs: CI now needs network access and
credentials to a stateful service to decide whether a pull request may merge; the
reference can change without any commit, so a build can start failing with no diff to
point at; and reproducing a six-month-old verdict depends on that server still holding
that run.

**B. Artifact in object storage** (S3 or GCS, keyed by branch). Cheaper than a server
and no UI to maintain. Same core problem as A: the reference is mutable and invisible in
review.

**C. A JSON file committed next to the code** (`baselines/baseline.json`).

## Decision

Option C, the boring choice.

The deciding property is reviewability. When someone re-records the baseline, the diff
appears in the pull request: the reviewer sees recall move from 0.8621 to 0.8636 and can
ask why, in the same place they are already looking. Blessing a new baseline becomes a
reviewed act rather than a UI click by whoever noticed the gate was red. It also means
the gate has no runtime dependencies: `pip install`, then `ragate gate`, and a fork with
no secrets can run it.

## Consequences

- The baseline file is roughly 100 KB of per-query detail for 140 queries. That is
  committed deliberately: per-query scores are what make the blame table possible, and
  the diff shows precisely which queries moved.
- History lives in git, not in a metrics UI. `git log baselines/baseline.json` is the
  trend view. There is no chart of recall over time, and that is a real loss.
- Trigger to revisit: more than one evaluation profile per repository, or a need to
  compare more than about 20 historical runs. At that point the baseline file becomes a
  pointer to an MLflow run id, and the gate gains a fetch step and the credentials that
  come with it.
- Because the file carries a fingerprint (corpus paths and k), the gate refuses to
  compare across a golden-set change instead of silently reporting a fake regression.
