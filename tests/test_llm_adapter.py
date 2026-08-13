"""Tests for the LLM reranker adapter's pure logic.

The request path is never exercised anywhere in this repository, because the build
environment has no credentials for the API, and no number in the README comes from it.
What can be tested without a key is the part most likely to be wrong in production:
parsing a model's reply, and what happens when that reply is malformed.
"""

from __future__ import annotations

import pytest

from ragate.errors import RagateError
from ragate.rerank.llm import LlmReranker, UsageCounter


def parse(text: str, valid=frozenset({1, 2, 3})):
    return LlmReranker._parse_permutation(text, set(valid))


def test_plain_array_is_parsed():
    assert parse("[3, 1, 2]") == [3, 1, 2]


def test_prose_around_the_array_is_tolerated():
    assert parse("Sure! Here you go:\n```json\n[2, 3]\n```\nHope that helps") == [2, 3]


def test_ids_outside_the_candidate_set_are_dropped():
    """A model inventing an id must not be able to inject a document that retrieval never
    returned."""
    assert parse("[99, 2, 1]") == [2, 1]


def test_repeated_ids_are_collapsed():
    assert parse("[2, 2, 1]") == [2, 1]


def test_non_integer_entries_are_skipped():
    assert parse('[2, "banana", 1]') == [2, 1]


@pytest.mark.parametrize("text", ["no array here", "[]", "[99]", "{}"])
def test_unusable_replies_raise(text):
    with pytest.raises(ValueError):
        parse(text)


def test_constructor_refuses_to_start_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RagateError, match="ANTHROPIC_API_KEY"):
        LlmReranker()


def test_usage_cost_is_unknown_rather_than_zero_without_prices():
    """Reporting unknown cost as free is how a FinOps number becomes a lie."""
    counter = UsageCounter(calls=2, input_tokens=1000, output_tokens=100)
    assert counter.cost() is None


def test_usage_cost_uses_the_supplied_price_table():
    counter = UsageCounter(
        input_tokens=1_000_000, output_tokens=500_000,
        prices_per_million={"input": 0.80, "output": 4.00},
    )
    assert counter.cost() == pytest.approx(0.80 + 2.00)
