"""Tests for the receipts checker itself.

This file exists because the checker was the one piece of this repository with no tests, and
it was also the one piece that reported success while three of the claims it guarded were
false. The first test below is the exact defect: a metric whose measured value is 11, a
document that says 10, and a check that used to pass.

The checker is a script rather than a package module, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load("check_readme_numbers")
collector = _load("collect_metrics")


def _registry(**metrics) -> dict:
    return {"metrics": metrics}


def _metric(value, display, anchors=None, source="test") -> dict:
    return {
        "value": value,
        "display": display if isinstance(display, list) else [display],
        "source": source,
        "note": "",
        "reproducible": True,
        "anchors": anchors,
    }


def test_an_anchored_count_fails_when_the_sentence_quotes_a_different_number():
    """The original defect: 11 measured, 10 written, and a bare substring search passes."""
    document = {"README.md": "status FAIL, with 10 queries named in the blame table.\n"}
    registry = _registry(
        regressed=_metric(11, "11", ["{} queries named in the blame table"])
    )

    # Unanchored, the old behaviour: "11" is not in this text, but in a real README of a
    # few hundred lines it always is, which is what made the check worthless.
    assert "11" not in document["README.md"]

    problems, checked, phrases = checker.check(registry, document)

    assert checked == 1
    assert phrases == 1
    assert len(problems) == 1
    assert "11 queries named in the blame table" in problems[0]
    # The message must show what the document actually says, not only what was expected.
    assert "says 10" in problems[0] or "10 queries named" in problems[0]


def test_an_anchored_count_passes_when_the_sentence_agrees():
    document = {"README.md": "status FAIL, with 11 queries named in the blame table.\n"}
    registry = _registry(
        regressed=_metric(11, "11", ["{} queries named in the blame table"])
    )

    problems, _, phrases = checker.check(registry, document)

    assert problems == []
    assert phrases == 1


def test_a_short_value_with_no_anchor_is_itself_a_failure():
    """A count that declares no location cannot be verified, so it must not pass."""
    document = {"README.md": "the suite has 170 tests and 11 of something else\n"}
    registry = _registry(tests=_metric(170, "170", None))

    problems, _, _ = checker.check(registry, document)

    assert len(problems) == 1
    assert "no anchor" in problems[0]
    assert "ANCHORS" in problems[0]


def test_a_long_value_with_no_anchor_is_still_searched_loosely():
    """0.8762 is specific enough that finding it anywhere is evidence."""
    registry = _registry(recall=_metric(0.8762, "0.8762", None))

    assert checker.check(registry, {"README.md": "recall@5 0.8762 on the candidate\n"})[0] == []
    problems, _, _ = checker.check(registry, {"README.md": "recall@5 0.9001\n"})
    assert len(problems) == 1
    assert "no checked document contains" in problems[0]


def test_every_anchor_must_be_satisfied_not_just_one():
    """The stale '159 tests' survived because another 170 existed elsewhere in the file."""
    document = {"README.md": "badge/tests-170-blue\nmake verify  # lint, 159 tests, and more\n"}
    registry = _registry(
        tests=_metric(170, "170", ["badge/tests-{}-", "lint, {} tests, and"])
    )

    problems, _, phrases = checker.check(registry, document)

    assert phrases == 2
    assert len(problems) == 1
    assert "lint, 170 tests, and" in problems[0]


def test_a_claim_in_any_checked_document_counts():
    """Two of the three stale numbers were not in the README at all."""
    documents = {
        "README.md": "nothing measured here\n",
        "ragate.yaml": "# Held-out recall@5 0.9070 without it\n",
    }
    registry = _registry(without=_metric(0.907, "0.9070", ["recall@5 {} without it"]))

    assert checker.check(registry, documents)[0] == []


def test_any_display_variant_may_satisfy_an_anchor():
    document = {"README.md": "92% line coverage\n"}
    registry = _registry(coverage=_metric(92.1, ["92%", "92.1%"], ["{} line coverage"]))

    assert checker.check(registry, document)[0] == []


def test_the_checker_refuses_a_document_list_it_cannot_read(tmp_path: Path):
    with pytest.raises(SystemExit, match="do not exist"):
        checker.load_documents([str(tmp_path / "absent.md")])


def test_the_registry_cannot_anchor_a_metric_that_was_never_measured():
    """An anchor for a deleted metric is a check that silently stops running."""
    with pytest.raises(SystemExit, match="did not measure"):
        collector.attach_anchors({"something_else": _metric(1, "1")})


def test_every_anchor_in_the_registry_names_a_real_metric_and_uses_one_placeholder():
    for name, templates in collector.ANCHORS.items():
        assert templates, f"{name} has an empty anchor list"
        for template in templates:
            assert template.count("{}") == 1, f"{name}: {template!r} needs exactly one {{}}"


# Deliberately not tested here: "the committed docs/metrics.json is fully anchored". That
# assertion would be circular, because the registry is regenerated by the same command that
# runs this suite, so a fresh anchor could never be added without one red run in between.
# The property is enforced where it belongs, by the checker itself in the readme-receipts
# CI job, which fails on any short unanchored value.
