"""LLM reranker adapter (Anthropic messages API).

Status, stated plainly: this code has never been executed. The build environment for
this repository has no credentials for the API, so there are no latency, cost, or
quality numbers for it anywhere in this project, and none are claimed. It is here
because the reranking boundary should not assume a linear model, and because the
interface a hosted reranker needs (batching, a strict output contract, a bounded retry
policy, a cost counter) is a design decision worth making before it is needed rather
than after.

The default reranker is the linear model in model.py, which is trained and measured on
the golden set with out-of-fold evaluation. See docs/adr/ADR-005.

Contract choices that matter if this is ever switched on:
  * The model is asked for a permutation of candidate ids, not for scores. Scores from
    an LLM are not comparable across calls, so they cannot be thresholded or fused.
  * Any candidate the model omits keeps its retrieval rank, appended after the ones it
    did rank. A reranker that can silently drop documents is a retrieval bug generator.
  * Every call's input and output token counts are accumulated, so cost per 1,000
    queries is measured rather than estimated.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..corpus import IndexUnit
from ..errors import RagateError
from ..logging_setup import get_logger

log = get_logger(__name__)

_MAX_ATTEMPTS = 4
_PROMPT = """You are ranking candidate support articles for a user's question.

Question: {query}

Candidates:
{candidates}

Return only a JSON array of candidate ids, most relevant first, including every id
exactly once. No prose, no explanation."""


@dataclass
class UsageCounter:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    prices_per_million: dict[str, float] = field(default_factory=dict)

    def cost(self) -> float | None:
        """Total cost, or None when no price table was supplied.

        Returning None rather than 0.0 on purpose: a missing price table is unknown
        cost, and reporting unknown cost as free is how a FinOps number becomes a lie.
        """
        if not self.prices_per_million:
            return None
        return (
            self.input_tokens / 1e6 * self.prices_per_million.get("input", 0.0)
            + self.output_tokens / 1e6 * self.prices_per_million.get("output", 0.0)
        )


class LlmReranker:
    name = "llm"

    def __init__(self, model: str = "claude-haiku-4-5", max_candidates: int = 20,
                 prices_per_million: dict[str, float] | None = None) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RagateError(
                "ANTHROPIC_API_KEY is not set; the linear reranker needs no credentials"
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RagateError("install the llm extra: pip install '.[llm]'") from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_candidates = max_candidates
        self.usage = UsageCounter(prices_per_million=prices_per_million or {})

    @staticmethod
    def _parse_permutation(text: str, valid: set[int]) -> list[int]:
        """Pull the id array out of the reply, tolerating a code fence or stray prose."""
        match = re.search(r"\[[^\]]*\]", text, flags=re.S)
        if match is None:
            raise ValueError("no JSON array in reply")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, list):
            raise ValueError("reply is not a list")
        seen: set[int] = set()
        order: list[int] = []
        for item in parsed:
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value in valid and value not in seen:
                seen.add(value)
                order.append(value)
        if not order:
            raise ValueError("reply contained no valid candidate ids")
        return order

    def rerank(
        self, query: str, candidates: list[tuple[int, float]], units: Sequence[IndexUnit]
    ) -> list[tuple[int, float]]:
        window = candidates[: self.max_candidates]
        tail = candidates[self.max_candidates :]
        if not window:
            return candidates
        listing = "\n".join(
            f"[{unit_index}] {units[unit_index].text[:400]}" for unit_index, _ in window
        )
        prompt = _PROMPT.format(query=query, candidates=listing)
        valid = {unit_index for unit_index, _ in window}

        delay = 0.5
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.usage.calls += 1
                self.usage.input_tokens += response.usage.input_tokens
                self.usage.output_tokens += response.usage.output_tokens
                order = self._parse_permutation(response.content[0].text, valid)
                break
            except Exception as exc:
                self.usage.retries += 1
                if attempt == _MAX_ATTEMPTS:
                    # Falling back to retrieval order rather than failing the run: a
                    # reranker outage should degrade quality, not break the gate.
                    log.warning(
                        "llm rerank failed, keeping retrieval order",
                        extra={"attempts": attempt, "error": str(exc)},
                    )
                    return candidates
                time.sleep(delay)
                delay *= 2

        scored = {unit_index: score for unit_index, score in window}
        ranked = [(unit_index, scored[unit_index]) for unit_index in order]
        # Anything the model left out keeps its retrieval position, after the ranked set.
        omitted = [item for item in window if item[0] not in set(order)]
        return ranked + omitted + tail
