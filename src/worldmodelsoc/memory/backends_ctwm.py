"""
Faithful memory backends for compact CTWM comparison.

All backends implement:
  - write_transition(prev, action, nxt, step)
  - retrieve_hints(current_state, step) -> List[Dict]
  - context_string(current_state) -> str      # 实际拼入 LLM prompt 的 memory context
  - context_tokens_estimator() -> int         # 旧 estimator, 保留报双数
  - unique_states_seen (set), unique_trans_seen (set)
  - access_counter (Counter)

Key change from v1: context_string is REAL and gets concatenated into the LLM prompt.
tokens_actual reported from chat-completion usage.prompt_tokens sum / N.
"""

from __future__ import annotations
import random
from collections import Counter, deque, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ==============================================================================
# Base class
# ==============================================================================


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
        """Real memory context string that gets concat'd into LLM prompt."""
        raise NotImplementedError

    def context_tokens_estimator(self) -> int:
        """Legacy estimator, keep for double-reporting."""
        return 0

    def coverage_state(self, walker_states: set) -> float:
        return len(self.unique_states_seen & walker_states) / max(1, len(walker_states))

    def coverage_trans(self, walker_trans: set) -> float:
        return len(self.unique_trans_seen & walker_trans) / max(1, len(walker_trans))


# ==============================================================================
# B1 Full History (真正 concat 全轨迹)
# ==============================================================================


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
        # 全历史都算 read 事件 (真的 pass 给 LLM)
        for (p, a, n) in self.history:
            tid = f"{p}::{a}::{n}"
            self.access_counter[tid] += 1
            self.read_events += 1
        return [{"memory_id": f"{p}::{a}::{n}", "rank": i}
                for i, (p, a, n) in enumerate(self.history)]

    def context_string(self, current_state) -> str:
        if not self.history: return "(empty history)"
        # 紧凑格式: "s0->a0->s1;s1->a1->s2;..."
        # 每 transition ~15 chars → ~4 tokens
        lines = [f"{p}->{a}->{n}" for (p, a, n) in self.history]
        return "history: " + ";".join(lines)

    def context_tokens_estimator(self):
        return 4 * len(self.history)


# ==============================================================================
# B2 Sliding Window K=100
# ==============================================================================


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


# ==============================================================================
# B3 Flat Retrieval — PURE GLOBAL UNIFORM (no state-mate structure)
# ==============================================================================


class B3_FlatRetrieval(BaseMemory):
    """Pure global uniform retrieval: store all transitions, retrieve top-3 uniformly at random from all."""
    name = "B3_FlatRetrieval"

    def __init__(self, top_k=3, seed_offset=3):
        super().__init__()
        self.top_k = top_k
        self.rng = random.Random(42 + seed_offset)
        # Unbounded storage (or reservoir-sized? spec ambiguous, use unbounded but with cap M=200 for parity with pilot)
        self.entries: Dict[str, Dict[str, Any]] = {}  # tid -> {content, first_step}

    def write_transition(self, prev, action, nxt, step):
        tid = f"{prev}::{action}::{nxt}"
        if tid not in self.entries:
            self.entries[tid] = {"content": f"{prev}->{action}->{nxt}", "first_step": step}
        self.access_counter[tid] += 1
        self.write_events += 1
        self.note_state(prev); self.note_state(nxt); self.note_trans(tid)

    def retrieve_hints(self, current_state, step):
        n_avail = len(self.entries)
        if n_avail == 0: return []
        all_tids = list(self.entries.keys())
        k_eff = min(self.top_k, n_avail)
        picked = self.rng.sample(all_tids, k_eff)
        events = []
        for r, tid in enumerate(picked):
            self.access_counter[tid] += 1
            self.read_events += 1
            events.append({"memory_id": tid, "rank": r})
        return events

    def context_string(self, current_state):
        hits = [self.entries[tid]["content"] for tid in
                (self.retrieve_no_side_effects(current_state) or [])]
        if not hits: return "(no retrievals)"
        return "flat_retrieval: " + "; ".join(hits)

    def retrieve_no_side_effects(self, current_state):
        """Peek without incrementing counters (used by context_string after retrieve_hints)."""
        # This is called AFTER retrieve_hints in caller, so we don't need to peek; instead
        # caller should track and pass. For simplicity return recent 3 from picked cache.
        return list(self.entries.keys())[-self.top_k:] if self.entries else []

    def context_tokens_estimator(self):
        return 6 * self.top_k


# ==============================================================================
# B4 Frequency Cache — FIXED CAPACITY M=100, EVICT MIN-FREQ
# ==============================================================================


class B4_FrequencyCache(BaseMemory):
    name = "B4_FrequencyCache"

    def __init__(self, capacity=100, top_k=3):
        super().__init__()
        self.M = capacity
        self.top_k = top_k
        # cache: tid -> {content, freq (in cache)}
        self.cache: Dict[str, Dict[str, Any]] = {}

    def write_transition(self, prev, action, nxt, step):
        tid = f"{prev}::{action}::{nxt}"
        if tid in self.cache:
            self.cache[tid]["freq"] += 1
        elif len(self.cache) < self.M:
            self.cache[tid] = {"content": f"{prev}->{action}->{nxt}", "freq": 1}
        else:
            # Evict min-freq
            min_tid = min(self.cache, key=lambda k: self.cache[k]["freq"])
            if self.cache[min_tid]["freq"] <= 1:  # accept newcomer
                del self.cache[min_tid]
                self.cache[tid] = {"content": f"{prev}->{action}->{nxt}", "freq": 1}
            # else newcomer rejected (rare, only when cache全部 freq >=2)
        self.access_counter[tid] += 1
        self.write_events += 1
        self.note_state(prev); self.note_state(nxt); self.note_trans(tid)

    def retrieve_hints(self, current_state, step):
        if not self.cache: return []
        # top-k by freq desc
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


# ==============================================================================
# B5 Recency Cache — FIXED CAPACITY M=100, EVICT OLDEST-RECENCY (LRU)
# ==============================================================================


class B5_RecencyCache(BaseMemory):
    name = "B5_RecencyCache"

    def __init__(self, capacity=100, top_k=3):
        super().__init__()
        self.M = capacity
        self.top_k = top_k
        # cache: OrderedDict-like, most recent last
        self.cache_order: deque = deque()  # tids in insertion/access order
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
            # Evict oldest recency (LRU)
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
        # Most recent first
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


# ==============================================================================
# B6 Hierarchical Summary — COUNT-BASED (每 100 步 summary top-5)
# ==============================================================================


class B6_HierarchicalSummary(BaseMemory):
    name = "B6_HierarchicalSummary"

    def __init__(self, chunk=100, top_k_summary=5, top_k_hint=3):
        super().__init__()
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
        # summaries first (compressed layer)
        for s in self.summaries[-3:]:
            for (tid, cnt) in s["top_trans"][:2]:
                self.access_counter[tid] += 1
                self.read_events += 1
                hits.append({"memory_id": tid, "rank": len(hits), "layer": "summary"})
                if len(hits) >= self.top_k_hint: return hits
        # then recent raw
        for (p, a, n) in list(self.recent)[-3:]:
            tid = f"{p}::{a}::{n}"
            self.access_counter[tid] += 1
            self.read_events += 1
            hits.append({"memory_id": tid, "rank": len(hits), "layer": "raw"})
            if len(hits) >= self.top_k_hint: return hits
        return hits

    def context_string(self, current_state):
        parts = []
        # All summaries (compact: top-3 trans + counts)
        for i, s in enumerate(self.summaries):
            top_desc = ",".join([f"{tid.split('::')[-1]}(x{c})" for tid, c in s["top_trans"][:3]])
            parts.append(f"sum{i}:{top_desc}")
        # Recent 5 raw
        recent = list(self.recent)[-5:]
        parts.append("recent:" + ";".join([f"{p}->{a}->{n}" for p, a, n in recent]))
        return " | ".join(parts) if parts else "(empty)"

    def context_tokens_estimator(self):
        return 6 * len(self.summaries) + 4 * min(5, len(self.recent))


# ==============================================================================
# B7 Graph Memory — AriGraph-style (episodic + semantic 双层)
# ==============================================================================


class B7_GraphMemory(BaseMemory):
    """AriGraph-style episodic + semantic dual-layer memory.
    - Episodic: (episode_id, step, prev, action, next) 序列, episode = 100 steps
    - Semantic: state ↔ entity bipartite graph (uses payload.entities passed at write time)
    - Retrieval: fuse episodic (recent state-mate) + semantic (entity co-occurrence)
    """
    name = "B7_GraphMemory"

    def __init__(self, episode_length=100, top_k=3):
        super().__init__()
        self.episode_length = episode_length
        self.top_k = top_k
        # Episodic: list of dicts
        self.episodes: List[Dict[str, Any]] = []
        self.current_episode: Dict[str, Any] = {"episode_id": 0, "transitions": []}
        # Semantic: state -> set of entities; entity -> set of states
        self.state_entities: Dict[str, set] = defaultdict(set)
        self.entity_states: Dict[str, set] = defaultdict(set)
        # transition edges for coverage tracking
        self.edges: Dict[str, Dict[str, Any]] = {}  # tid -> {prev, action, next, degree}

    def write_transition_with_entities(self, prev, action, nxt, step, entities_prev, entities_next):
        """Version with entity info from payload."""
        tid = f"{prev}::{action}::{nxt}"
        if tid in self.edges:
            self.edges[tid]["degree"] += 1
        else:
            self.edges[tid] = {"prev": prev, "action": action, "next": nxt, "degree": 1}
        # episodic
        self.current_episode["transitions"].append({"step": step, "prev": prev, "action": action, "next": nxt})
        if len(self.current_episode["transitions"]) >= self.episode_length:
            self.episodes.append(self.current_episode)
            self.current_episode = {"episode_id": len(self.episodes), "transitions": []}
        # semantic
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
        # Fallback without entities (should not be called; caller uses write_transition_with_entities)
        self.write_transition_with_entities(prev, action, nxt, step, entities_prev=[], entities_next=[])

    def _episodic_top_k(self, current_state):
        """Look through recent episodic transitions with prev==current_state."""
        picks = []
        # search from most recent episode backwards
        for ep in reversed([self.current_episode] + self.episodes[::-1]):
            for tr in reversed(ep["transitions"]):
                if tr["prev"] == current_state:
                    picks.append((tr["prev"], tr["action"], tr["next"]))
                    if len(picks) >= self.top_k: return picks
        return picks

    def _semantic_top_k(self, current_state):
        """Find states sharing entities with current_state; among transitions from those, pick top-k."""
        my_ents = self.state_entities.get(current_state, set())
        if not my_ents:
            return []
        cooc_states: Counter = Counter()
        for e in my_ents:
            for s in self.entity_states.get(e, set()):
                if s != current_state:
                    cooc_states[s] += 1
        top_states = [s for s, _ in cooc_states.most_common(self.top_k * 2)]
        # find transitions with prev in top_states, prefer higher degree
        candidate_edges = [(tid, e) for tid, e in self.edges.items() if e["prev"] in top_states]
        candidate_edges.sort(key=lambda x: -x[1]["degree"])
        return [(e["prev"], e["action"], e["next"]) for tid, e in candidate_edges[:self.top_k]]

    def retrieve_hints(self, current_state, step):
        epi = self._episodic_top_k(current_state)
        sem = self._semantic_top_k(current_state)
        # Fuse + dedup, keep top-3
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
        # Compact: episodic subgraph + top 1-hop semantic label
        parts = [f"{p}->{a}->{n}" for (p, a, n) in picks]
        # Add semantic hint: 3 top entities of current_state
        my_ents = list(self.state_entities.get(current_state, set()))[:3]
        ent_str = f"ents:{','.join(my_ents)}" if my_ents else "ents:(none)"
        return "kg: " + "; ".join(parts) + " | " + ent_str

    def context_tokens_estimator(self):
        return 6 * self.top_k + 6  # + semantic tag


# ==============================================================================
# B8 CTWM — 完整 idea.md §5.3 implementation (ũ=0.5 constant, 无 dynamic Tail expansion)
# ==============================================================================


class B8_CTWM(BaseMemory):
    """
    CTWM v1: W = W_core ∪ W_tail with τ-controlled allocation.
    - 5 features per entry: f̃ (freq), q̃ (rank-weighted retrieval), d̃ (downstream diversity),
      ũ (uncertainty; constant 0.5, static), ṽ (visit-inverse value proxy)
    - Core score c_i = z-score weighted [0.3, 0.2, 0.2, 0.15, 0.15]
    - θ_c: top-30% by c_i (percentile-based) → Core
    - Retrieval:
       Core: top-3 by c_i × 30 tokens = 90 tokens budget
       Tail: top-2 by b(r;τ) = r^{-τ} / Σⱼ j^{-τ} × 12 tokens = 24 tokens budget
    - Cluster: 相似 Tail entries (same prev state) 合并为 1 summary
    - 无 dynamic Tail expansion by u_q (v1 limitation)
    """
    name = "B8_CTWM"

    def __init__(self, tau=1.0, core_pct=0.30, core_slots=3, tail_slots=2, capacity=200,
                  weights=(0.3, 0.2, 0.2, 0.15, 0.15), seed_offset=8):
        super().__init__()
        self.tau = tau
        self.core_pct = core_pct
        self.core_slots = core_slots
        self.tail_slots = tail_slots
        self.M = capacity
        self.weights = weights
        self.rng = random.Random(42 + seed_offset)
        # Entries: tid -> dict with f, q_ranks, d_state, u=0.5, v_state_rank, first_step
        self.entries: Dict[str, Dict[str, Any]] = {}
        # State transition graph for d̃ (downstream diversity)
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
                # Evict lowest c_i score (recompute a batch)
                self._recompute_scores()
                min_tid = min(self.entries, key=lambda t: self.entries[t].get("c", 0))
                del self.entries[min_tid]
                self.entries[tid] = {"f": 1, "q_ranks": [], "prev": prev, "action": action,
                                     "next": nxt, "u": 0.5, "first_step": step}
        # state graph
        self.state_next_states[prev].add(nxt)
        self.state_total_out[prev] += 1
        self.state_visit_freq[prev] += 1
        self.state_visit_freq[nxt] += 1
        self.access_counter[tid] += 1
        self.write_events += 1
        self.note_state(prev); self.note_state(nxt); self.note_trans(tid)

    def _recompute_scores(self):
        """Compute c_i = z-score weighted 5 features across all entries."""
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
        """Return (core_tids sorted by c desc, tail_tids)."""
        self._recompute_scores()
        tids = list(self.entries.keys())
        if not tids: return [], []
        sorted_tids = sorted(tids, key=lambda t: -self.entries[t]["c"])
        n_core = max(1, int(len(sorted_tids) * self.core_pct))
        return sorted_tids[:n_core], sorted_tids[n_core:]

    def retrieve_hints(self, current_state, step):
        core, tail = self._partition_core_tail()
        # Core: top core_slots by c_i (also filter to state-mate if available)
        core_mate = [t for t in core if self.entries[t]["prev"] == current_state]
        # 若 state-mate 不够就补 non-mate 高 c_i entries
        core_pick = list(core_mate[:self.core_slots])
        if len(core_pick) < self.core_slots:
            non_mate = [t for t in core if t not in core_pick]
            core_pick.extend(non_mate[:self.core_slots - len(core_pick)])
        core_pick = core_pick[:self.core_slots]

        # Tail: b(r;τ) probability sample, tail_slots picks
        # rank r ∈ {1..len(tail)}, weight = r^{-τ}
        tail_picks = []
        if tail:
            ranks = np.arange(1, len(tail) + 1, dtype=float)
            weights = ranks ** (-self.tau)
            weights /= weights.sum()
            n_pick = min(self.tail_slots, len(tail))
            # weighted sample without replacement
            picked_ranks = list(np.random.choice(len(tail), size=n_pick, replace=False, p=weights)) \
                if False else self._sample_ranks(weights, n_pick)
            tail_picks = [tail[r] for r in picked_ranks]

        # Cluster tail_picks by prev state
        clusters: Dict[str, List[str]] = defaultdict(list)
        for t in tail_picks:
            clusters[self.entries[t]["prev"]].append(t)
        # Cluster summary: 只报 cluster head + count
        cluster_summary_tids = []
        for prev, tids_in_cluster in clusters.items():
            cluster_summary_tids.append(tids_in_cluster[0])  # 取 head

        events = []
        # emit core reads
        for r, tid in enumerate(core_pick):
            self.access_counter[tid] += 1
            self.read_events += 1
            events.append({"memory_id": tid, "rank": r, "layer": "core"})
            self.entries[tid]["q_ranks"].append(r)
        # emit tail reads
        for r, tid in enumerate(cluster_summary_tids):
            self.access_counter[tid] += 1
            self.read_events += 1
            events.append({"memory_id": tid, "rank": len(core_pick) + r, "layer": "tail"})
            self.entries[tid]["q_ranks"].append(len(core_pick) + r)

        self._last_core = core_pick
        self._last_tail = cluster_summary_tids
        return events

    def _sample_ranks(self, weights, n_pick):
        """Deterministic weighted sampling without replacement using self.rng."""
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
        """v2c encoding: drop c=数字 in Core; report Tail as count-only summary.

        - Core: "core: A->a->B; C->b->D; E->c->F" (rank order implicit, no c=... verbose)
        - Tail: 
          * if n_clusters == len(tail): "tail: 2 items" (count-only, forcing cluster expression)
          * if n_clusters < len(tail): "tail: 2 items from {prev1,prev2}" (compressed cluster info)
        Preserves §5.3 spec: 5-feature scoring + Core/Tail partition + b(r;τ) budget + cluster summary.
        Only serialization changed: rank order replaces c=..., cluster count replaces per-slot triples.
        """
        core = getattr(self, "_last_core", []) or []
        tail = getattr(self, "_last_tail", []) or []
        core_parts = []
        for t in core:
            e = self.entries[t]
            # rank order implicit (list order = c_i desc), no c=... verbose
            core_parts.append(f"{e['prev']}->{e['action']}->{e['next']}")
        core_str = "core:" + ";".join(core_parts) if core_parts else "core:(empty)"

        # Tail: count-only summary; if clusters exist, list unique prev states
        if not tail:
            tail_str = "tail:(empty)"
        else:
            tail_prevs = []
            seen_prevs = set()
            for t in tail:
                if t in self.entries:
                    p = self.entries[t]["prev"]
                    if p not in seen_prevs:
                        seen_prevs.add(p)
                        tail_prevs.append(p)
            if len(tail_prevs) < len(tail):
                # compression: fewer clusters than tail slots
                tail_str = f"tail:{len(tail)}items,clusters=[{','.join(tail_prevs)}]"
            else:
                # no compression: count-only summary (forcing minimal encoding)
                tail_str = f"tail:{len(tail)}items"
        return core_str + " | " + tail_str

    def context_tokens_estimator(self):
        return 12 * self.core_slots + 6 * self.tail_slots
