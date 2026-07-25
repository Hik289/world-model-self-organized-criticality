"""Reservoir-style memory utilities shared by the experiment scripts."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Dict, List

import numpy as np
from scipy import stats as spstats


def gini(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    x = np.sort(np.asarray(x, dtype=np.float64))
    if x.sum() == 0:
        return 0.0
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def summary_stats(freqs: List[int]) -> Dict[str, float]:
    arr = np.array(sorted(freqs, reverse=True), dtype=np.int64)
    if arr.size == 0:
        return {
            "n_unique": 0,
            "top1": 0,
            "median": 0.0,
            "max_over_median": 0.0,
            "skew": 0.0,
            "gini": 0.0,
            "top10pct_share": 0.0,
            "singleton_fraction": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "total": 0,
        }
    total = int(arr.sum())
    med = float(np.median(arr))
    top10n = max(1, int(np.ceil(arr.size * 0.1)))
    return {
        "n_unique": int(arr.size),
        "top1": int(arr[0]),
        "median": med,
        "max_over_median": float(arr[0] / med) if med > 0 else float("inf"),
        "skew": (
            float(spstats.skew(arr))
            if arr.size > 1 and np.ptp(arr) > 0
            else 0.0
        ),
        "gini": gini(arr),
        "top10pct_share": float(arr[:top10n].sum()) / max(1, total),
        "singleton_fraction": float((arr == 1).sum()) / arr.size,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "total": total,
    }


def pl_fit(freqs: List[int]) -> Dict[str, Any]:
    """Small, dependency-light fit summary used by tau sweeps."""
    arr = np.asarray([x for x in freqs if x > 0], dtype=float)
    if arr.size < 2:
        return {"alpha_hat": None, "xmin_hat": None, "lognormal_sigma": None}
    xmin = float(max(1.0, np.percentile(arr, 10)))
    tail = arr[arr >= xmin]
    if tail.size < 2:
        tail = arr
        xmin = float(arr.min())
    denom = np.sum(np.log(tail / max(xmin, 1e-12)))
    alpha = 1.0 + len(tail) / denom if denom > 1e-12 else float("inf")
    positive = arr[arr > 0]
    return {
        "alpha_hat": float(alpha) if math.isfinite(alpha) else None,
        "xmin_hat": xmin,
        "lognormal_sigma": float(np.std(np.log(positive))) if positive.size else None,
    }


class StateAwareReservoirMemory:
    """Bounded transition memory with state-aware retrieval.

    The interface matches the original experiment scripts:
    `write(...)` and `retrieve(...)` both return event dictionaries while
    updating an access counter used for heavy-tail analysis.
    """

    def __init__(self, capacity: int = 200, rng: random.Random | None = None):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self.rng = rng or random.Random(42)
        self.slots: Dict[str, Dict[str, Any]] = {}
        self.access_counter: Counter = Counter()
        self.write_events = 0
        self.read_events = 0
        self.evict_events = 0
        self.conflicts = 0

    def _evict_if_needed(self) -> None:
        if len(self.slots) < self.capacity:
            return
        min_freq = min(rec["freq"] for rec in self.slots.values())
        candidates = [mid for mid, rec in self.slots.items() if rec["freq"] == min_freq]
        victim = self.rng.choice(candidates)
        del self.slots[victim]
        self.evict_events += 1

    def write(self, memory_id: str, content: str, prev: str, action: str,
              nxt: str, step: int) -> List[Dict[str, Any]]:
        access_kind = "overwrite" if memory_id in self.slots else "write"
        if memory_id not in self.slots:
            self._evict_if_needed()
            self.slots[memory_id] = {
                "content": content,
                "prev": prev,
                "action": action,
                "next": nxt,
                "freq": 0,
                "first_seen_step": step,
            }
        else:
            rec = self.slots[memory_id]
            if rec.get("content") != content:
                self.conflicts += 1
        rec = self.slots[memory_id]
        rec["freq"] += 1
        rec["last_seen_step"] = step
        self.access_counter[memory_id] += 1
        self.write_events += 1
        return [{"memory_id": memory_id, "access_kind": access_kind, "freq": rec["freq"]}]

    def retrieve(self, current_state: str, k: int = 3, step: int = 0) -> List[Dict[str, Any]]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if not self.slots:
            return []
        state_mates = [
            (mid, rec) for mid, rec in self.slots.items()
            if rec.get("prev") == current_state or rec.get("next") == current_state
        ]
        pool = state_mates if state_mates else list(self.slots.items())
        pool = sorted(pool, key=lambda item: (-item[1].get("freq", 0), item[0]))
        events = []
        for rank, (mid, rec) in enumerate(pool[:k]):
            rec["last_seen_step"] = step
            self.access_counter[mid] += 1
            self.read_events += 1
            events.append({
                "memory_id": mid,
                "access_kind": "read",
                "retrieval_rank": rank,
                "content": rec.get("content", ""),
                "access_freq_running": self.access_counter[mid],
            })
        return events


class TauReservoirMemory(StateAwareReservoirMemory):
    """State-aware reservoir with tau-controlled rank sampling."""

    def __init__(self, capacity: int = 200, rng: random.Random | None = None,
                 tau: float = 1.0, K_pool: int = 10, M_pass: int = 3):
        super().__init__(capacity=capacity, rng=rng)
        if not math.isfinite(tau) or tau < 0:
            raise ValueError("tau must be non-negative")
        if K_pool < 1:
            raise ValueError("K_pool must be at least 1")
        if M_pass < 1:
            raise ValueError("M_pass must be at least 1")
        self.tau = tau
        self.K_pool = K_pool
        self.M_pass = M_pass

    def retrieve(self, current_state: str, step: int = 0) -> List[Dict[str, Any]]:  # type: ignore[override]
        if not self.slots:
            return []
        state_mates = [
            (mid, rec) for mid, rec in self.slots.items()
            if rec.get("prev") == current_state or rec.get("next") == current_state
        ]
        pool = state_mates if state_mates else list(self.slots.items())
        pool = sorted(pool, key=lambda item: (-item[1].get("freq", 0), item[0]))[:self.K_pool]
        if not pool:
            return []
        ranks = np.arange(1, len(pool) + 1, dtype=float)
        weights = ranks ** (-self.tau)
        weights = weights / weights.sum()
        n_pick = min(self.M_pass, len(pool))
        remaining = list(range(len(pool)))
        remaining_weights = weights.tolist()
        seen = []
        for _ in range(n_pick):
            selected = self.rng.choices(
                range(len(remaining)),
                weights=remaining_weights,
                k=1,
            )[0]
            seen.append(remaining.pop(selected))
            remaining_weights.pop(selected)
        events = []
        for rank, idx in enumerate(seen[:n_pick]):
            mid, rec = pool[idx]
            rec["last_seen_step"] = step
            self.access_counter[mid] += 1
            self.read_events += 1
            events.append({
                "memory_id": mid,
                "access_kind": "read",
                "retrieval_rank": rank,
                "content": rec.get("content", ""),
                "access_freq_running": self.access_counter[mid],
            })
        return events
