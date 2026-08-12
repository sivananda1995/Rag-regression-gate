"""Configuration loading.

Precedence, lowest to highest: dataclass defaults, YAML profile, environment
variables. Nothing in this package reads os.environ or a file path directly; all
of it arrives through a Config instance, which keeps the evaluation reproducible
and makes every knob visible in one place.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

ENV_PREFIX = "RAGATE"


def _sha256(path: str | Path) -> str:
    """Short content hash of an evaluation input, or "missing" if it is absent.

    Absent is not an error here: the loaders raise a clear CorpusError later, and a
    fingerprint call should not be the thing that reports a missing file.
    """
    p = Path(path)
    if not p.exists():
        return "missing"
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


@dataclass
class CorpusConfig:
    path: str = "data/corpus.jsonl"
    queries: str = "data/golden_queries.jsonl"


@dataclass
class ChunkingConfig:
    strategy: str = "sentence_window"
    target_chars: int = 480
    overlap_chars: int = 120
    # Index byte-identical chunk text once, mapped to every document it appears in.
    dedupe_identical: bool = True

    def validate(self) -> None:
        if self.strategy not in {"sentence_window", "fixed"}:
            raise ConfigError(f"unknown chunking strategy: {self.strategy}")
        if self.target_chars <= 0:
            raise ConfigError("chunking.target_chars must be positive")
        if self.overlap_chars < 0:
            raise ConfigError("chunking.overlap_chars must not be negative")
        if self.overlap_chars >= self.target_chars:
            raise ConfigError(
                "chunking.overlap_chars must be smaller than target_chars, "
                f"got overlap={self.overlap_chars} target={self.target_chars}"
            )


@dataclass
class EmbedderConfig:
    provider: str = "hashing"
    dimensions: int = 512
    idf_weighting: bool = True
    char_ngram: int = 4
    model: str = ""

    def validate(self) -> None:
        if self.provider not in {"hashing", "openai"}:
            raise ConfigError(f"unknown embedder provider: {self.provider}")
        if self.dimensions < 16:
            raise ConfigError("embedder.dimensions must be at least 16")
        if self.char_ngram < 2:
            raise ConfigError("embedder.char_ngram must be at least 2")


@dataclass
class IndexConfig:
    backend: str = "flat"
    metric: str = "cosine"

    def validate(self) -> None:
        if self.backend not in {"flat", "faiss"}:
            raise ConfigError(f"unknown index backend: {self.backend}")
        if self.metric != "cosine":
            raise ConfigError("only cosine similarity is implemented")


@dataclass
class EvaluateConfig:
    k: int = 5

    def validate(self) -> None:
        if self.k <= 0:
            raise ConfigError("evaluate.k must be positive")


@dataclass
class GateConfig:
    primary_metric: str = "recall_at_k"
    max_absolute_drop: float = 0.02
    bootstrap_confidence: float = 0.95
    bootstrap_resamples: int = 5000
    blame_threshold: float = 0.5
    baseline_path: str = "baselines/baseline.json"

    def validate(self) -> None:
        if not 0.0 <= self.max_absolute_drop < 1.0:
            raise ConfigError("gate.max_absolute_drop must be in [0, 1)")
        if not 0.5 < self.bootstrap_confidence < 1.0:
            raise ConfigError("gate.bootstrap_confidence must be in (0.5, 1)")
        if self.bootstrap_resamples < 100:
            raise ConfigError("gate.bootstrap_resamples must be at least 100")


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"


@dataclass
class Config:
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> Config:
        for f in fields(self):
            section = getattr(self, f.name)
            validator = getattr(section, "validate", None)
            if validator is not None:
                validator()
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> dict[str, Any]:
        """Identity of the evaluation set, stored with every report.

        The gate refuses to compare two reports whose fingerprints differ, because a
        metric measured on a different golden set is not a comparison.

        Keyed on file content, not on file paths. An earlier version used the paths,
        which got the dangerous case backwards: editing the corpus in place left the
        fingerprint unchanged and the baseline silently stale, while moving a file to a
        new directory invalidated a baseline that was still perfectly valid.
        """
        return {
            "corpus_sha256": _sha256(self.corpus.path),
            "queries_sha256": _sha256(self.corpus.queries),
            "k": self.evaluate.k,
        }


def _coerce(current: Any, raw: str) -> Any:
    if isinstance(current, bool):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(f"cannot read {raw!r} as a boolean")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def load(path: str | Path | None = "ragate.yaml", env: dict[str, str] | None = None) -> Config:
    """Build a validated Config from a YAML profile plus environment overrides."""
    env = os.environ if env is None else env
    config = Config()

    if path is not None:
        p = Path(path)
        if p.exists():
            try:
                raw = yaml.safe_load(p.read_text()) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"{p} is not valid YAML: {exc}") from exc
            if not isinstance(raw, dict):
                raise ConfigError(f"{p} must contain a mapping at the top level")
            for section_name, values in raw.items():
                if not hasattr(config, section_name):
                    raise ConfigError(f"unknown config section: {section_name}")
                section = getattr(config, section_name)
                if not isinstance(values, dict):
                    raise ConfigError(f"config section {section_name} must be a mapping")
                for key, value in values.items():
                    if not hasattr(section, key):
                        raise ConfigError(f"unknown config key: {section_name}.{key}")
                    setattr(section, key, value)

    for f in fields(config):
        section = getattr(config, f.name)
        for sf in fields(section):
            env_key = f"{ENV_PREFIX}_{f.name.upper()}_{sf.name.upper()}"
            if env_key in env:
                setattr(section, sf.name, _coerce(getattr(section, sf.name), env[env_key]))

    return config.validate()
