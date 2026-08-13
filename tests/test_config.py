from __future__ import annotations

import pytest

from ragate.config import load
from ragate.errors import ConfigError


def _profile(tmp_path, body: str):
    path = tmp_path / "ragate.yaml"
    path.write_text(body)
    return path


def test_defaults_apply_when_no_profile_exists(tmp_path):
    cfg = load(tmp_path / "absent.yaml", env={})
    assert cfg.evaluate.k == 5
    assert cfg.embedder.provider == "hashing"


def test_profile_overrides_defaults(tmp_path):
    cfg = load(_profile(tmp_path, "evaluate:\n  k: 10\n"), env={})
    assert cfg.evaluate.k == 10


def test_env_overrides_profile_and_coerces_types(tmp_path):
    path = _profile(tmp_path, "evaluate:\n  k: 10\ngate:\n  max_absolute_drop: 0.05\n")
    cfg = load(
        path,
        env={
            "RAGATE_EVALUATE_K": "3",
            "RAGATE_GATE_MAX_ABSOLUTE_DROP": "0.01",
            "RAGATE_EMBEDDER_IDF_WEIGHTING": "false",
        },
    )
    assert cfg.evaluate.k == 3
    assert cfg.gate.max_absolute_drop == pytest.approx(0.01)
    assert cfg.embedder.idf_weighting is False


def test_unknown_section_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown config section"):
        load(_profile(tmp_path, "nonsense:\n  a: 1\n"), env={})


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown config key"):
        load(_profile(tmp_path, "evaluate:\n  kk: 1\n"), env={})


def test_invalid_yaml_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not valid YAML"):
        load(_profile(tmp_path, "evaluate:\n  k: [1,\n"), env={})


def test_unparseable_boolean_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="boolean"):
        load(_profile(tmp_path, ""), env={"RAGATE_EMBEDDER_IDF_WEIGHTING": "perhaps"})


@pytest.mark.parametrize(
    "body, message",
    [
        ("evaluate:\n  k: 0\n", "k must be positive"),
        ("gate:\n  max_absolute_drop: 1.5\n", "max_absolute_drop"),
        ("gate:\n  bootstrap_resamples: 10\n", "bootstrap_resamples"),
        ("embedder:\n  provider: magic\n", "unknown embedder provider"),
        ("index:\n  backend: annoy\n", "unknown index backend"),
        ("chunking:\n  strategy: vibes\n", "unknown chunking strategy"),
    ],
)
def test_validation_rejects_impossible_values(tmp_path, body, message):
    with pytest.raises(ConfigError, match=message):
        load(_profile(tmp_path, body), env={})


def test_fingerprint_is_keyed_on_content_not_paths(tmp_path):
    """Editing the corpus in place must invalidate a baseline; moving the file must not."""
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"doc_id": "a", "title": "t", "text": "one"}\n')
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"query_id": "q", "text": "one", "relevant_doc_ids": ["a"]}\n')

    def fingerprint_for(corpus_path, queries_path):
        body = (
            f"corpus:\n  path: {corpus_path}\n  queries: {queries_path}\nevaluate:\n  k: 4\n"
        )
        return load(_profile(tmp_path, body), env={}).fingerprint()

    original = fingerprint_for(corpus, queries)
    assert original["k"] == 4
    assert "embedder" not in original

    moved = tmp_path / "elsewhere.jsonl"
    moved.write_bytes(corpus.read_bytes())
    assert fingerprint_for(moved, queries) == original

    corpus.write_text('{"doc_id": "a", "title": "t", "text": "one edited"}\n')
    assert fingerprint_for(corpus, queries) != original


def test_fingerprint_tolerates_a_missing_file(tmp_path):
    """A missing corpus is reported by the loader with a line-accurate message; the
    fingerprint must not be the thing that raises first."""
    body = f"corpus:\n  path: {tmp_path / 'absent.jsonl'}\n  queries: {tmp_path / 'no.jsonl'}\n"
    fingerprint = load(_profile(tmp_path, body), env={}).fingerprint()
    assert fingerprint["corpus_sha256"] == "missing"
    assert fingerprint["queries_sha256"] == "missing"


def test_extends_inherits_the_parent_and_overlays_the_child(tmp_path):
    parent = tmp_path / "base.yaml"
    parent.write_text(
        "evaluate:\n  k: 5\nretriever:\n  mode: bm25\n  bm25_k1: 1.5\nrerank:\n  enabled: true\n"
    )
    child = tmp_path / "candidate.yaml"
    child.write_text("extends: base.yaml\nretriever:\n  bm25_k1: 0.9\n")
    cfg = load(child, env={})
    # Overlaid key changes, sibling keys in the same section survive, other sections too.
    assert cfg.retriever.bm25_k1 == pytest.approx(0.9)
    assert cfg.retriever.mode == "bm25"
    assert cfg.evaluate.k == 5
    assert cfg.rerank.enabled is True


def test_extends_resolves_relative_to_the_child_file(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "ragate.yaml").write_text("evaluate:\n  k: 7\n")
    child = tmp_path / "configs" / "candidate.yaml"
    child.write_text("extends: ../ragate.yaml\nevaluate:\n  splits_path: ''\n")
    cfg = load(child, env={})
    assert cfg.evaluate.k == 7
    assert cfg.evaluate.splits_path == ""


def test_env_still_wins_over_an_inherited_value(tmp_path):
    (tmp_path / "base.yaml").write_text("evaluate:\n  k: 5\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\n")
    assert load(child, env={"RAGATE_EVALUATE_K": "2"}).evaluate.k == 2


def test_missing_extends_target_is_reported(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text("extends: nowhere.yaml\n")
    with pytest.raises(ConfigError, match="extends target does not exist"):
        load(child, env={})


def test_non_string_extends_is_refused(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text("extends:\n  - a.yaml\n")
    with pytest.raises(ConfigError, match="extends must be a path string"):
        load(child, env={})


def test_an_extends_cycle_is_stopped(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("extends: b.yaml\n")
    b.write_text("extends: a.yaml\n")
    with pytest.raises(ConfigError, match="extends each other|deeper than"):
        load(a, env={})


def test_committed_candidate_profiles_all_load():
    """Every profile shipped in configs/ has to be valid, or a README instruction is a
    lie. Cheap test, catches a stale key after a config rename."""
    from pathlib import Path

    profiles = sorted(Path("configs").glob("*.yaml"))
    assert profiles, "expected candidate profiles in configs/"
    for profile in profiles:
        load(profile).validate()
