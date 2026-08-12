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


def test_fingerprint_covers_corpus_and_k_only(tmp_path):
    cfg = load(_profile(tmp_path, "evaluate:\n  k: 4\n"), env={})
    fingerprint = cfg.fingerprint()
    assert fingerprint["k"] == 4
    assert "corpus" in fingerprint
    assert "embedder" not in fingerprint
