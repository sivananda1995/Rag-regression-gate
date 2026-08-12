"""Exception hierarchy for ragate.

Every failure surfaced to the CLI is one of these, so the CLI can map a failure
class to a process exit code without inspecting messages.
"""


class RagateError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(RagateError):
    """Configuration is missing, malformed, or internally inconsistent."""


class CorpusError(RagateError):
    """Corpus or golden-query file is unreadable or violates its schema."""


class EmbedderError(RagateError):
    """An embedding provider failed or is misconfigured."""


class BaselineError(RagateError):
    """The baseline artifact is missing or was produced by an incompatible setup."""
