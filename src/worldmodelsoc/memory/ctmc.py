from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import tiktoken


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    text: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


def rank_weights(size: int, tau: float) -> np.ndarray:
    if size < 0 or not math.isfinite(tau) or tau < 0:
        raise ValueError("size and tau must be non-negative and tau must be finite")
    if size == 0:
        return np.empty(0, dtype=float)
    logits = -tau * np.log(np.arange(1, size + 1, dtype=float))
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def integer_shares(weights: np.ndarray, budget: int) -> np.ndarray:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if not len(weights):
        return np.empty(0, dtype=int)
    values = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and have positive mass")
    exact = budget * values / values.sum()
    shares = np.floor(exact).astype(int)
    order = np.argsort(-(exact - shares), kind="stable")
    shares[order[:budget - int(shares.sum())]] += 1
    return shares


class TokenCodec:
    def __init__(self, encoding: str = "cl100k_base"):
        self.encoding_name = encoding
        self.encoder = tiktoken.get_encoding(encoding)

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text, disallowed_special=())

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def clip(self, text: str, budget: int) -> str:
        if budget <= 0:
            return ""
        tokens = self.encode(text)[:budget]
        result = self.encoder.decode(tokens)
        while self.count(result) > budget:
            tokens = tokens[:-1]
            result = self.encoder.decode(tokens)
        return result


def record_text(record: MemoryRecord, rank: int, encoding: str) -> str:
    if encoding == "compact":
        return f"rank {rank}: {record.text}"
    if encoding == "verbose":
        return (
            f"Retrieved memory {record.memory_id}; retrieval rank {rank}; "
            f"retrieval score {record.score:.6f}; metadata {record.metadata}; "
            f"memory content: {record.text}"
        )
    raise ValueError(f"unknown serialization: {encoding}")


def transition_summary(records: list[MemoryRecord], budget: int, codec: TokenCodec) -> str:
    groups = Counter()
    for record in records:
        meta = record.metadata
        groups[(meta.get("prev", "?"), meta.get("action", "?"), meta.get("next", "?"))] += meta.get("count", 1)
    dominant = sorted(groups, key=lambda key: (-groups[key], key))[:3]
    rare = sorted(groups, key=lambda key: (groups[key], key))[:3]
    text = f"count={len(records)}; dominant transitions=" + "; ".join(
        f"{prev}->{action}->{nxt} ({groups[(prev, action, nxt)]})"
        for prev, action, nxt in dominant
    )
    text += "; rare exceptions=" + "; ".join(
        f"{prev}->{action}->{nxt} ({groups[(prev, action, nxt)]})"
        for prev, action, nxt in rare if (prev, action, nxt) not in dominant
    )
    return codec.clip(text, budget)


def pack_context(
    records: list[MemoryRecord],
    method: str,
    encoding: str,
    budget: int,
    core_size: int,
    tau: float,
    codec: TokenCodec,
    summarize: Callable[[list[MemoryRecord], int, TokenCodec], str] = transition_summary,
) -> tuple[str, dict]:
    if budget < 64 or core_size < 1:
        raise ValueError("memory budget must be at least 64 and core_size must be positive")
    if encoding not in ("compact", "verbose"):
        raise ValueError("encoding must be compact or verbose")
    ids = [record.memory_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("ranked memory identifiers must be unique")
    weights = rank_weights(len(records), tau)
    metadata = {
        "ranked_memory_ids": ids, "method": method, "serialization": encoding,
        "memory_budget": budget, "tokenizer": codec.encoding_name, "tau": tau,
        "allocation_weights": weights.tolist(),
    }
    if method == "full_history":
        text = "\n".join(record_text(record, rank, encoding) for rank, record in enumerate(records, 1))
        metadata.update(memory_tokens=codec.count(text), budget_applied=False)
        return text, metadata
    if method in ("graph_memory", "flat_retrieval"):
        text = codec.clip("\n".join(record_text(record, rank, encoding) for rank, record in enumerate(records[:core_size], 1)), budget)
        metadata.update(memory_tokens=codec.count(text), budget_applied=True, selected_memory_ids=ids[:core_size])
        return text, metadata
    if method != "ctmc":
        raise ValueError(f"unknown method: {method}")
    if not records:
        metadata.update(memory_tokens=0, core_mass=0.0, core_size=0, core_budget=0, tail_budget=0)
        return "", metadata
    k = min(core_size, len(records) - 1) if len(records) > 1 else 1
    mass = float(weights[:k].sum())
    available = budget - codec.count("Core memories:\n\nTail summary:\n") - 8
    core_budget, tail_budget = integer_shares(np.array([mass, max(0.0, 1.0 - mass)]), available)
    core_shares = integer_shares(weights[:k], max(0, int(core_budget) - k))
    core = "\n".join(
        codec.clip(record_text(record, rank, encoding), int(share))
        for rank, (record, share) in enumerate(zip(records[:k], core_shares), 1)
    )
    tail = summarize(records[k:], int(tail_budget), codec) if records[k:] and tail_budget else ""
    text = codec.clip(f"Core memories:\n{core}\nTail summary:\n{tail}", budget)
    metadata.update(
        memory_tokens=codec.count(text), budget_applied=True, core_size=k, core_mass=mass,
        core_budget=int(core_budget), tail_budget=int(tail_budget),
        core_item_budgets=core_shares.tolist(), core_memory_ids=ids[:k], tail_memory_ids=ids[k:],
    )
    return text, metadata


def bm25_rank(records: list[MemoryRecord], query: str) -> list[MemoryRecord]:
    if not records:
        return []
    tokenize = lambda text: re.findall(r"\w+", text.casefold())
    terms = set(tokenize(query))
    documents = [Counter(tokenize(record.text)) for record in records]
    lengths = np.array([sum(doc.values()) for doc in documents], dtype=float)
    average = max(1.0, float(lengths.mean()))
    frequencies = Counter(term for doc in documents for term in terms if term in doc)
    ranked = []
    for record, doc, length in zip(records, documents, lengths):
        score = 0.0
        for term in terms:
            frequency = doc[term]
            inverse = math.log1p((len(documents) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
            score += inverse * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / average))
        values = asdict(record)
        values["score"] = float(score)
        ranked.append(MemoryRecord(**values))
    return sorted(ranked, key=lambda record: (-record.score, record.memory_id))
