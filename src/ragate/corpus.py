"""Corpus loading, golden-query loading, and chunking.

Chunking lives here rather than inside the embedder because a chunking change is
one of the two most common causes of a silent retrieval regression (the other is a
model or prompt change), and the gate needs to attribute a drop to it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import ChunkingConfig
from .errors import CorpusError
from .logging_setup import get_logger

log = get_logger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class IndexUnit:
    """One vector's worth of text, and every document that text belongs to.

    Templated knowledge bases repeat whole passages across articles: on the corpus in
    this repository 2,422 of 3,780 chunks are byte-identical to another chunk. Indexing
    each copy separately buys nothing (identical text scores identically) and costs
    three things: memory, query time, and graph quality in an approximate index, where
    duplicate points distort neighbour selection badly enough to cut retrieved-score
    parity to 0.81. Indexing the text once and carrying the list of documents it came
    from removes all three costs and leaves document-level metrics unchanged.
    """

    text: str
    doc_ids: tuple[str, ...]
    occurrences: int


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    relevant_doc_ids: tuple[str, ...]


def _read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise CorpusError(f"file not found: {p}")
    rows: list[dict] = []
    with p.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{p}:{lineno} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise CorpusError(f"{p}:{lineno} must be a JSON object")
            rows.append(row)
    if not rows:
        raise CorpusError(f"{p} contains no records")
    return rows


def load_documents(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    seen: set[str] = set()
    for row in _read_jsonl(path):
        for key in ("doc_id", "title", "text"):
            if key not in row:
                raise CorpusError(f"document record missing required field {key!r}: {row}")
        doc_id = str(row["doc_id"])
        if doc_id in seen:
            raise CorpusError(f"duplicate doc_id in corpus: {doc_id}")
        seen.add(doc_id)
        docs.append(Document(doc_id=doc_id, title=str(row["title"]), text=str(row["text"])))
    log.info("corpus loaded", extra={"documents": len(docs), "path": str(path)})
    return docs


def load_queries(path: str | Path, known_doc_ids: set[str] | None = None) -> list[Query]:
    queries: list[Query] = []
    for row in _read_jsonl(path):
        for key in ("query_id", "text", "relevant_doc_ids"):
            if key not in row:
                raise CorpusError(f"query record missing required field {key!r}: {row}")
        relevant = row["relevant_doc_ids"]
        if not isinstance(relevant, list) or not relevant:
            raise CorpusError(f"query {row['query_id']} must list at least one relevant doc_id")
        if known_doc_ids is not None:
            unknown = [d for d in relevant if d not in known_doc_ids]
            if unknown:
                # A label pointing at a deleted document silently depresses recall
                # forever, so it is a hard error rather than a warning.
                raise CorpusError(
                    f"query {row['query_id']} labels doc_ids absent from the corpus: {unknown}"
                )
        queries.append(
            Query(
                query_id=str(row["query_id"]),
                text=str(row["text"]),
                relevant_doc_ids=tuple(str(d) for d in relevant),
            )
        )
    log.info("golden set loaded", extra={"queries": len(queries), "path": str(path)})
    return queries


def _fixed_chunks(text: str, target: int, overlap: int) -> list[str]:
    step = target - overlap
    out = []
    for start in range(0, max(len(text), 1), step):
        piece = text[start : start + target]
        if piece.strip():
            out.append(piece)
        if start + target >= len(text):
            break
    return out


def _sentence_window_chunks(text: str, target: int, overlap: int) -> list[str]:
    """Pack whole sentences up to target_chars, then carry the tail of the previous
    chunk forward as overlap so an answer spanning a sentence boundary is not split
    away from its context."""
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    if not sentences:
        return []
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        if current and size + len(sentence) + 1 > target:
            chunks.append(" ".join(current))
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev) + 1
            current = carry
            size = carried
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def collapse_duplicate_chunks(chunks: list[Chunk]) -> list[IndexUnit]:
    """Group chunks by exact text, preserving first-seen order.

    Order is preserved so that the index, and therefore any tie between equally
    scoring units, is deterministic across runs.
    """
    grouped: dict[str, list[str]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.text, []).append(chunk.doc_id)
    units = [
        IndexUnit(text=text, doc_ids=tuple(doc_ids), occurrences=len(doc_ids))
        for text, doc_ids in grouped.items()
    ]
    duplicates = len(chunks) - len(units)
    log.info(
        "duplicate chunks collapsed",
        extra={
            "chunks": len(chunks),
            "index_units": len(units),
            "duplicates_removed": duplicates,
            "reduction_pct": round(100.0 * duplicates / max(len(chunks), 1), 1),
        },
    )
    return units


def build_index_units(chunks: list[Chunk], dedupe: bool) -> list[IndexUnit]:
    if dedupe:
        return collapse_duplicate_chunks(chunks)
    return [IndexUnit(text=c.text, doc_ids=(c.doc_id,), occurrences=1) for c in chunks]


def chunk_documents(docs: list[Document], cfg: ChunkingConfig) -> list[Chunk]:
    cfg.validate()
    chunker = _sentence_window_chunks if cfg.strategy == "sentence_window" else _fixed_chunks
    chunks: list[Chunk] = []
    for doc in docs:
        pieces = chunker(doc.full_text, cfg.target_chars, cfg.overlap_chars)
        if not pieces:
            raise CorpusError(f"document {doc.doc_id} produced no chunks")
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(chunk_id=f"{doc.doc_id}#{i}", doc_id=doc.doc_id, text=piece))
    log.info(
        "documents chunked",
        extra={
            "strategy": cfg.strategy,
            "documents": len(docs),
            "chunks": len(chunks),
            "chunks_per_doc": round(len(chunks) / len(docs), 2),
        },
    )
    return chunks
