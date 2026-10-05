from __future__ import annotations
import math
import random
import re
from collections import Counter, deque, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import tiktoken


class BaseMemory:
    name = "Base"

    def __init__(self):
        self.access_counter: Counter = Counter()
        self.unique_states_seen: set = set()
        self.unique_trans_seen: set = set()
        self.write_events = 0
        self.read_events = 0

    def note_state(self, sid): self.unique_states_seen.add(sid)
    def note_trans(self, tid): self.unique_trans_seen.add(tid)

    def write_transition(self, prev, action, nxt, step):
        raise NotImplementedError

    def retrieve_hints(self, current_state, step) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def context_string(self, current_state) -> str:
        raise NotImplementedError

    def context_tokens_estimator(self) -> int:
        return 0

    def retained_transition_ids(self) -> set:
        return set(self.unique_trans_seen)

    def retained_state_ids(self) -> set:
        states = set()
        for tid in self.retained_transition_ids():
            parts = tid.split("::")
            if len(parts) == 3:
                states.update((parts[0], parts[2]))
        return states

    def coverage_state(self, walker_states: set) -> float:
        return len(self.retained_state_ids() & walker_states) / max(1, len(walker_states))

    def coverage_trans(self, walker_trans: set) -> float:
        return len(self.retained_transition_ids() & walker_trans) / max(1, len(walker_trans))


class B1_FullHistory(BaseMemory):
    name = "B1_FullHistory"

    def __init__(self):
        super().__init__()
        self.history: List[Tuple[str, str, str]] = []

    def write_transition(self, prev, action, nxt, step):
        self.history.append((prev, action, nxt))
        tid = f"{prev}::{action}::{nxt}"
        self.access_counter[tid] += 1
        self.write_events += 1
        self.note_state(prev); self.note_state(nxt); self.note_trans(tid)

    def retrieve_hints(self, current_state, step):
        for (p, a, n) in self.history:
            tid = f"{p}::{a}::{n}"
            self.access_counter[tid] += 1
            self.read_events += 1
        return [{"memory_id": f"{p}::{a}::{n}", "rank": i}
                for i, (p, a, n) in enumerate(self.history)]

    def context_string(self, current_state) -> str:
        if not self.history: return "(empty history)"
        lines = [f"{p}->{a}->{n}" for (p, a, n) in self.history]
        return "history: " + ";".join(lines)

    def context_tokens_estimator(self):
        return 4 * len(self.history)


class B2_SlidingWindow(BaseMemory):
    name = "B2_SlidingWindow"

    def __init__(self, K=100):
        super().__init__()
        self.K = K
        self.window: deque = deque(maxlen=K)

    def write_transition(self, prev, action, nxt, step):
        self.window.append((prev, action, nxt))
        tid = f"{prev}::{action}::{nxt}"
        self.access_counter[tid] += 1
        self.write_events += 1
        self.note_state(prev); self.note_state(nxt); self.note_trans(tid)

    def retrieve_hints(self, current_state, step):
        for (p, a, n) in self.window:
            tid = f"{p}::{a}::{n}"
            self.access_counter[tid] += 1
            self.read_events += 1
        return [{"memory_id": f"{p}::{a}::{n}", "rank": i}
                for i, (p, a, n) in enumerate(list(self.window))]

    def context_string(self, current_state):
        if not self.window: return "(empty window)"
        lines = [f"{p}->{a}->{n}" for (p, a, n) in self.window]
        return "window: " + ";".join(lines)

    def context_tokens_estimator(self):
        return 4 * len(self.window)

    def retained_transition_ids(self):
        return {f"{p}::{a}::{n}" for p, a, n in self.window}


class B3_FlatRetrieval(BaseMemory):
    name = "B3_FlatRetrieval"

    def __init__(self, top_k=3, seed=42, seed_offset=3):
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.top_k = top_k
        self.rng = random.Random(seed + seed_offset)
        self.entries: Dict[str, Dict[str, Any]] = {}
        self._last_retrieved: List[str] = []

    def write_transition(self, prev, action, nxt, step):
        tid = f"{prev}::{action}::{nxt}"
        if tid not in self.entries:
            self.entries[tid] = {"content": f"{prev}->{action}->{nxt}", "first_step": step}
        self.access_counter[tid] += 1
        self.write_events += 1
        self.note_state(prev); self.note_state(nxt); self.note_trans(tid)

    def retrieve_hints(self, current_state, step):
        n_avail = len(self.entries)
        if n_avail == 0:
            self._last_retrieved = []
            return []
        all_tids = list(self.entries.keys())
        k_eff = min(self.top_k, n_avail)
        picked = self.rng.sample(all_tids, k_eff)
        self._last_retrieved = picked
        events = []
        for r, tid in enumerate(picked):
            self.access_counter[tid] += 1
            self.read_events += 1
            events.append({"memory_id": tid, "rank": r})
        return events

    def context_string(self, current_state):
        hits = [self.entries[tid]["content"] for tid in self._last_retrieved]
        if not hits: return "(no retrievals)"
        return "flat_retrieval: " + "; ".join(hits)

    def retrieve_no_side_effects(self, current_state):
        return list(self._last_retrieved)

    def context_tokens_estimator(self):
        return 6 * self.top_k

    def retained_transition_ids(self):
        return set(self.entries)


class B4_FrequencyCache(BaseMemory):
    name = "B4_FrequencyCache"

    def __init__(self, capacity=100, top_k=3):
        super().__init__()
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.M = capacity
        self.top_k = top_k
        self.cache: Dict[str, Dict[str, Any]] = {}

    def write_transition(self, prev, action, nxt, step):
        tid = f"{prev}::{action}::{nxt}"
        if tid in self.cache:
            self.cache[tid]["freq"] += 1
        elif len(self.cache) < self.M:
            self.cache[tid] = {"content": f"{prev}->{action}->{nxt}", "freq": 1}
        else:
            min_tid = min(self.cache, key=lambda k: self.cache[k]["freq"])
            if self.cache[min_tid]["freq"] <= 1:
                del self.cache[min_tid]
                self.cache[tid] = {"content": f"{prev}->{action}->{nxt}", "freq": 1}
        self.access_counter[tid] += 1
        self.write_events += 1
        self.note_state(prev); self.note_state(nxt); self.note_trans(tid)

    def retrieve_hints(self, current_state, step):
        if not self.cache: return []
        sorted_tids = sorted(self.cache.keys(), key=lambda t: -self.cache[t]["freq"])
        picked = sorted_tids[:self.top_k]
        events = []
        for r, tid in enumerate(picked):
            self.access_counter[tid] += 1
            self.read_events += 1
            events.append({"memory_id": tid, "rank": r})
        return events

    def context_string(self, current_state):
        if not self.cache: return "(empty freq cache)"
        sorted_tids = sorted(self.cache.keys(), key=lambda t: -self.cache[t]["freq"])[:self.top_k]
        parts = [f"{self.cache[t]['content']}(f={self.cache[t]['freq']})" for t in sorted_tids]
        return "freq_cache: " + "; ".join(parts)

    def context_tokens_estimator(self):
        return 8 * self.top_k

    def retained_transition_ids(self):
        return set(self.cache)


class B5_RecencyCache(BaseMemory):
    name = "B5_RecencyCache"

    def __init__(self, capacity=100, top_k=3):
        super().__init__()
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.M = capacity
        self.top_k = top_k
        self.cache_order: deque = deque()
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.last_access: Dict[str, int] = {}

    def _touch(self, tid, step):
        self.last_access[tid] = step
        try:
            self.cache_order.remove(tid)
        except ValueError:
            pass
        self.cache_order.append(tid)

    def write_transition(self, prev, action, nxt, step):
        tid = f"{prev}::{action}::{nxt}"
        if tid in self.cache:
            self._touch(tid, step)
        elif len(self.cache) < self.M:
            self.cache[tid] = {"content": f"{prev}->{action}->{nxt}"}
            self._touch(tid, step)
        else:
            oldest = self.cache_order.popleft()
            del self.cache[oldest]
            del self.last_access[oldest]
            self.cache[tid] = {"content": f"{prev}->{action}->{nxt}"}
            self._touch(tid, step)
        self.access_counter[tid] += 1
        self.write_events += 1
        self.note_state(prev); self.note_state(nxt); self.note_trans(tid)

    def retrieve_hints(self, current_state, step):
        if not self.cache: return []
        recent = list(self.cache_order)[::-1][:self.top_k]
        events = []
        for r, tid in enumerate(recent):
            self.access_counter[tid] += 1
            self.read_events += 1
            events.append({"memory_id": tid, "rank": r})
        return events

    def context_string(self, current_state):
        if not self.cache: return "(empty recency cache)"
        recent = list(self.cache_order)[::-1][:self.top_k]
        parts = [self.cache[t]["content"] for t in recent]
        return "recency_cache: " + "; ".join(parts)

    def context_tokens_estimator(self):
        return 6 * self.top_k

    def retained_transition_ids(self):
        return set(self.cache)


class B6_HierarchicalSummary(BaseMemory):
    name = "B6_HierarchicalSummary"

    def __init__(self, chunk=100, top_k_summary=5, top_k_hint=3):
        super().__init__()
        if chunk < 1:
            raise ValueError("chunk must be at least 1")
        if top_k_summary < 1 or top_k_hint < 1:
            raise ValueError("summary and hint sizes must be at least 1")
        self.chunk = chunk
        self.top_k_summary = top_k_summary
        self.top_k_hint = top_k_hint
        self.recent: deque = deque(maxlen=chunk)
        self.summaries: List[Dict[str, Any]] = []
        self._chunk_visits: Counter = Counter()
        self._chunk_trans: Counter = Counter()

    def write_transition(self, prev, action, nxt, step):
        tid = f"{prev}::{action}::{nxt}"
        self.recent.append((prev, action, nxt))
        self._chunk_visits[nxt] += 1
        self._chunk_trans[tid] += 1
        self.access_counter[tid] += 1
        self.write_events += 1
        self.note_state(prev); self.note_state(nxt); self.note_trans(tid)
        if (step + 1) % self.chunk == 0:
            self.summaries.append({
                "step_end": step,
                "top_states": self._chunk_visits.most_common(self.top_k_summary),
                "top_trans": self._chunk_trans.most_common(self.top_k_summary),
            })
            self._chunk_visits = Counter()
            self._chunk_trans = Counter()

    def retrieve_hints(self, current_state, step):
        hits = []
        for s in reversed(self.summaries[-3:]):
            for (tid, _count) in s["top_trans"][:2]:
                self.access_counter[tid] += 1
                self.read_events += 1
                hits.append({"memory_id": tid, "rank": len(hits), "layer": "summary"})
                if len(hits) >= self.top_k_hint: return hits
        for (p, a, n) in list(self.recent)[-3:]:
            tid = f"{p}::{a}::{n}"
            self.access_counter[tid] += 1
            self.read_events += 1
            hits.append({"memory_id": tid, "rank": len(hits), "layer": "raw"})
            if len(hits) >= self.top_k_hint: return hits
        return hits

    def context_string(self, current_state):
        parts = []
        for i, s in enumerate(self.summaries):
            top_desc = ",".join([f"{tid.split('::')[-1]}(x{c})" for tid, c in s["top_trans"][:3]])
            parts.append(f"sum{i}:{top_desc}")
        recent = list(self.recent)[-5:]
        parts.append("recent:" + ";".join([f"{p}->{a}->{n}" for p, a, n in recent]))
        return " | ".join(parts) if parts else "(empty)"

    def context_tokens_estimator(self):
        return 6 * len(self.summaries) + 4 * min(5, len(self.recent))

    def retained_transition_ids(self):
        retained = {
            tid
            for summary in self.summaries
            for tid, _count in summary["top_trans"]
        }
        retained.update(f"{p}::{a}::{n}" for p, a, n in self.recent)
        return retained


class B7_GraphMemory(BaseMemory):
    name = "B7_GraphMemory"

    def __init__(self, episode_length=100, top_k=3):
        super().__init__()
        if episode_length < 1:
            raise ValueError("episode_length must be at least 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.episode_length = episode_length
        self.top_k = top_k
        self.episodes: List[Dict[str, Any]] = []
        self.current_episode: Dict[str, Any] = {"episode_id": 0, "transitions": []}
        self.state_entities: Dict[str, set] = defaultdict(set)
        self.entity_states: Dict[str, set] = defaultdict(set)
        self.edges: Dict[str, Dict[str, Any]] = {}

    def write_transition_with_entities(self, prev, action, nxt, step, entities_prev, entities_next):
        tid = f"{prev}::{action}::{nxt}"
        if tid in self.edges:
            self.edges[tid]["degree"] += 1
        else:
            self.edges[tid] = {"prev": prev, "action": action, "next": nxt, "degree": 1}
        self.current_episode["transitions"].append({"step": step, "prev": prev, "action": action, "next": nxt})
        if len(self.current_episode["transitions"]) >= self.episode_length:
            self.episodes.append(self.current_episode)
            self.current_episode = {"episode_id": len(self.episodes), "transitions": []}
        for e in entities_prev:
            self.state_entities[prev].add(e)
            self.entity_states[e].add(prev)
        for e in entities_next:
            self.state_entities[nxt].add(e)
            self.entity_states[e].add(nxt)

        self.access_counter[tid] += 1
        self.write_events += 1
        self.note_state(prev); self.note_state(nxt); self.note_trans(tid)

    def write_transition(self, prev, action, nxt, step):
        self.write_transition_with_entities(prev, action, nxt, step, entities_prev=[], entities_next=[])

    def _episodic_top_k(self, current_state):
        picks = []
        for ep in [self.current_episode, *reversed(self.episodes)]:
            for tr in reversed(ep["transitions"]):
                if tr["prev"] == current_state:
                    picks.append((tr["prev"], tr["action"], tr["next"]))
                    if len(picks) >= self.top_k: return picks
        return picks

    def _semantic_top_k(self, current_state):
        my_ents = self.state_entities.get(current_state, set())
        if not my_ents:
            return []
        cooc_states: Counter = Counter()
        for e in sorted(my_ents):
            for s in sorted(self.entity_states.get(e, set())):
                if s != current_state:
                    cooc_states[s] += 1
        top_states = [s for s, _ in cooc_states.most_common(self.top_k * 2)]
        candidate_edges = [(tid, e) for tid, e in self.edges.items() if e["prev"] in top_states]
        candidate_edges.sort(key=lambda x: (-x[1]["degree"], x[0]))
        return [(e["prev"], e["action"], e["next"]) for tid, e in candidate_edges[:self.top_k]]

    def retrieve_hints(self, current_state, step):
        epi = self._episodic_top_k(current_state)
        sem = self._semantic_top_k(current_state)
        seen = set()
        fused = []
        for triple in epi + sem:
            key = f"{triple[0]}::{triple[1]}::{triple[2]}"
            if key not in seen:
                seen.add(key)
                fused.append(triple)
            if len(fused) >= self.top_k: break
        events = []
        for r, (p, a, n) in enumerate(fused):
            tid = f"{p}::{a}::{n}"
            self.access_counter[tid] += 1
            self.read_events += 1
            events.append({"memory_id": tid, "rank": r})
        self._last_retrieved = fused
        return events

    def context_string(self, current_state):
        picks = getattr(self, "_last_retrieved", None) or self._episodic_top_k(current_state)[:self.top_k]
        if not picks: return "(empty KG)"
        parts = [f"{p}->{a}->{n}" for (p, a, n) in picks]
        my_ents = sorted(self.state_entities.get(current_state, set()))[:3]
        ent_str = f"ents:{','.join(my_ents)}" if my_ents else "ents:(none)"
        return "kg: " + "; ".join(parts) + " | " + ent_str

    def context_tokens_estimator(self):
        return 6 * self.top_k + 6

    def retained_transition_ids(self):
        return set(self.edges)


class B8_CTWM(BaseMemory):
    name = "B8_CTWM"

    def __init__(self, tau=1.0, core_pct=0.30, core_slots=3, tail_slots=2, capacity=200,
                  weights=(0.3, 0.2, 0.2, 0.15, 0.15), seed=42, seed_offset=8):
        super().__init__()
        if not np.isfinite(tau) or tau < 0:
            raise ValueError("tau must be non-negative")
        if not np.isfinite(core_pct) or not 0 < core_pct <= 1:
            raise ValueError("core_pct must be in (0, 1]")
        if core_slots < 1 or tail_slots < 0:
            raise ValueError("core_slots must be positive and tail_slots non-negative")
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if len(weights) != 5 or not np.all(np.isfinite(weights)):
            raise ValueError("weights must contain five finite values")
        self.tau = tau
        self.core_pct = core_pct
        self.core_slots = core_slots
        self.tail_slots = tail_slots
        self.M = capacity
        self.weights = weights
        self.rng = random.Random(seed + seed_offset)
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.state_next_states: Dict[str, set] = defaultdict(set)
        self.state_total_out: Dict[str, int] = defaultdict(int)
        self.state_visit_freq: Counter = Counter()

    def write_transition(self, prev, action, nxt, step):
        tid = f"{prev}::{action}::{nxt}"
        if tid in self.entries:
            self.entries[tid]["f"] += 1
        else:
            if len(self.entries) < self.M:
                self.entries[tid] = {"f": 1, "q_ranks": [], "prev": prev, "action": action,
                                     "next": nxt, "u": 0.5, "first_step": step}
            else:
                self._recompute_scores()
                min_tid = min(self.entries, key=lambda t: self.entries[t].get("c", 0))
                del self.entries[min_tid]
                self.entries[tid] = {"f": 1, "q_ranks": [], "prev": prev, "action": action,
                                     "next": nxt, "u": 0.5, "first_step": step}
        self.state_next_states[prev].add(nxt)
        self.state_total_out[prev] += 1
        self.state_visit_freq[prev] += 1
        self.state_visit_freq[nxt] += 1
        self.access_counter[tid] += 1
        self.write_events += 1
        self.note_state(prev); self.note_state(nxt); self.note_trans(tid)

    def _recompute_scores(self):
        if not self.entries: return
        tids = list(self.entries.keys())
        f_arr = np.array([self.entries[t]["f"] for t in tids], dtype=float)
        q_arr = np.array([
            (1.0 / (1.0 + np.mean(self.entries[t]["q_ranks"]))) if self.entries[t]["q_ranks"] else 0.0
            for t in tids
        ], dtype=float)
        d_arr = np.array([len(self.state_next_states[self.entries[t]["prev"]]) /
                           max(1, self.state_total_out[self.entries[t]["prev"]])
                           for t in tids], dtype=float)
        u_arr = np.array([self.entries[t]["u"] for t in tids], dtype=float)
        v_arr = np.array([1.0 / max(1, self.state_visit_freq[self.entries[t]["next"]])
                           for t in tids], dtype=float)

        def zsc(x):
            mu, sd = x.mean(), x.std()
            if sd < 1e-9: return x - mu
            return (x - mu) / sd

        c = (self.weights[0] * zsc(f_arr) + self.weights[1] * zsc(q_arr) +
             self.weights[2] * zsc(d_arr) + self.weights[3] * zsc(u_arr) +
             self.weights[4] * zsc(v_arr))
        for i, t in enumerate(tids):
            self.entries[t]["c"] = float(c[i])

    def _partition_core_tail(self):
        self._recompute_scores()
        tids = list(self.entries.keys())
        if not tids: return [], []
        sorted_tids = sorted(tids, key=lambda t: (-self.entries[t]["c"], t))
        n_core = max(1, int(len(sorted_tids) * self.core_pct))
        return sorted_tids[:n_core], sorted_tids[n_core:]

    def retrieve_hints(self, current_state, step):
        core, tail = self._partition_core_tail()
        core_mate = [t for t in core if self.entries[t]["prev"] == current_state]
        core_pick = list(core_mate[:self.core_slots])
        if len(core_pick) < self.core_slots:
            non_mate = [t for t in core if t not in core_pick]
            core_pick.extend(non_mate[:self.core_slots - len(core_pick)])
        core_pick = core_pick[:self.core_slots]

        tail_picks = []
        if tail:
            ranks = np.arange(1, len(tail) + 1, dtype=float)
            weights = ranks ** (-self.tau)
            weights /= weights.sum()
            n_pick = min(self.tail_slots, len(tail))
            picked_ranks = self._sample_ranks(weights, n_pick)
            tail_picks = [tail[r] for r in picked_ranks]

        clusters: Dict[str, List[str]] = defaultdict(list)
        for t in tail_picks:
            clusters[self.entries[t]["prev"]].append(t)
        cluster_summary_tids = []
        for _prev, tids_in_cluster in clusters.items():
            cluster_summary_tids.append(tids_in_cluster[0])

        events = []
        for r, tid in enumerate(core_pick):
            self.access_counter[tid] += 1
            self.read_events += 1
            events.append({"memory_id": tid, "rank": r, "layer": "core"})
            self.entries[tid]["q_ranks"].append(r)
        for r, tid in enumerate(cluster_summary_tids):
            self.access_counter[tid] += 1
            self.read_events += 1
            events.append({"memory_id": tid, "rank": len(core_pick) + r, "layer": "tail"})
            self.entries[tid]["q_ranks"].append(len(core_pick) + r)

        self._last_core = core_pick
        self._last_tail = cluster_summary_tids
        return events

    def _sample_ranks(self, weights, n_pick):
        remaining = list(range(len(weights)))
        remaining_w = list(weights)
        picked = []
        for _ in range(n_pick):
            if not remaining: break
            total = sum(remaining_w)
            if total <= 0: break
            u = self.rng.random() * total
            cum = 0.0
            sel = 0
            for i, w in enumerate(remaining_w):
                cum += w
                if u < cum:
                    sel = i; break
            picked.append(remaining[sel])
            remaining.pop(sel); remaining_w.pop(sel)
        return picked

    def context_string(self, current_state):
        core = getattr(self, "_last_core", []) or []
        tail = getattr(self, "_last_tail", []) or []
        core_parts = []
        for t in core:
            e = self.entries[t]
            core_parts.append(f"{e['prev']}->{e['action']}->{e['next']}")
        core_str = "core:" + ";".join(core_parts) if core_parts else "core:(empty)"

        if not tail:
            tail_str = "tail:(empty)"
        else:
            tail_str = f"tail:{len(tail)}clusters"
        return core_str + " | " + tail_str

    def context_tokens_estimator(self):
        return 12 * self.core_slots + 6 * self.tail_slots

    def retained_transition_ids(self):
        return set(self.entries)


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


class TransitionStore:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.records = {}
        self.history = []
        self.visited_states = set()
        self.visited_transitions = set()

    def write(self, prev: str, action: str, nxt: str, step: int, entities: list[str]):
        key = f"{prev}::{action}::{nxt}"
        count = self.records[key].metadata["count"] + 1 if key in self.records else 1
        metadata = {"prev": prev, "action": action, "next": nxt, "step": step, "count": count, "entities": entities}
        record = MemoryRecord(key, f"{prev}->{action}->{nxt}", metadata=metadata)
        self.records[key] = record
        self.history.append(MemoryRecord(f"{step}:{key}", record.text, metadata=metadata))
        self.visited_states.update((prev, nxt))
        self.visited_transitions.add(key)
        if len(self.records) > self.capacity:
            victim = min(self.records, key=lambda item: (self.records[item].metadata["count"], self.records[item].metadata["step"], item))
            del self.records[victim]

    def rank(self, state: str, entities: list[str], method: str) -> list[MemoryRecord]:
        if method == "full_history":
            return list(self.history)
        if method == "flat_retrieval":
            return bm25_rank(list(self.records.values()), state + " " + " ".join(entities))
        ranked = []
        for record in self.records.values():
            meta = record.metadata
            score = 100.0 * (meta["prev"] == state)
            score += 10.0 * len(set(entities) & set(meta["entities"]))
            score += meta["count"] / (1.0 + meta["count"])
            ranked.append(MemoryRecord(record.memory_id, record.text, score, meta))
        return sorted(ranked, key=lambda record: (-record.score, record.memory_id))
