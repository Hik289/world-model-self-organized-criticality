"""
LLM policy study — intervene on the random-walk policy with an LLM walker.

Cleanroom setup (only policy differs from the scale-free random-walk setting):
- graph: scale_free |V|=100 seed=42
- memory: v2 StateAwareReservoirMemory (sort-by-freq + top_k=3, M=200)
- N: 由 sanity 决定, target ≥5000
- policy: OpenAI-compatible chat endpoint (key from env or .secrets/llm.key)

Prompt structure (简单, 保守 tokens):
  system: 你是一个 walker, 给你当前 state 信息和 top-3 memory hints, 选一个 action.
  user: current_state=v_XX (entities=..., actions=[a1, a2, ...])
        neighbors: [v_YY -> a1, v_ZZ -> a2, ...]
        recent_states: [v_.., v_..]  (last 5)
        memory hints: [tx_...(freq=..), ...]
        Choose one action index (0-based). Reply ONLY: {"action_idx": <int>}

fallback: JSON 解析失败或 idx 超范围 → uniform random (记入 fallback_count)

per-step 记录:
- api_call cost (prompt+completion tokens)
- action 是否 llm-picked or fallback
- decision timestamp

输出:
- results/scale_free_v100_s42_llm_meta.json (含 cost tally + all summary stats)
- results/scale_free_v100_s42_llm_mem_counts.json (memory-access counts)
- results/scale_free_v100_s42_llm_state_counts.json
- results/scale_free_v100_s42_llm_trans_counts.json
- results/scale_free_v100_s42_llm_mem_time.jsonl (memory-access 时序 for PSD)
- results/scale_free_v100_s42_llm_actions.jsonl (每步 policy 决策明细, cost audit 用)
- results/scale_free_v100_s42_llm_{state,mem}_{first,second}_half.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from scipy import stats as spstats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldmodelsoc.memory.reservoir import StateAwareReservoirMemory  # noqa: E402
from worldmodelsoc.env.synthetic_graph_world import build_graph, build_state_payloads  # noqa: E402
from worldmodelsoc.llm_config import LLM_API_BASE_URL, LLM_MODEL, make_openai_client  # noqa: E402


# ==============================================================================
# LLM client
# ==============================================================================

# Default pricing estimate; update for the model/provider you run.
PRICE_PROMPT_PER_1M = 0.15   # USD per 1M prompt tokens
PRICE_COMPL_PER_1M = 0.60    # USD per 1M completion tokens


def make_client() -> OpenAI:
    return make_openai_client()


# ==============================================================================
# Cost accounting
# ==============================================================================


class CostAccountant:
    def __init__(self, budget_usd: float = 5.0):
        self.tokens_prompt = 0
        self.tokens_completion = 0
        self.api_calls = 0
        self.fallback_count = 0
        self.budget_usd = budget_usd
        self.error_count = 0
        self.last_error: Optional[str] = None

    def add(self, prompt_toks: int, completion_toks: int):
        self.tokens_prompt += prompt_toks
        self.tokens_completion += completion_toks
        self.api_calls += 1

    def estimated_usd(self) -> float:
        return (self.tokens_prompt / 1e6) * PRICE_PROMPT_PER_1M \
             + (self.tokens_completion / 1e6) * PRICE_COMPL_PER_1M

    def budget_remaining_usd(self) -> float:
        return self.budget_usd - self.estimated_usd()

    def budget_exceeded(self) -> bool:
        return self.estimated_usd() >= self.budget_usd


# ==============================================================================
# LLM policy
# ==============================================================================


def _extract_action_idx(text: str, n_actions: int) -> Optional[int]:
    """Parse JSON {"action_idx": <int>} out of LLM response. Return None on failure."""
    if not text:
        return None
    m = re.search(r"\{[^{}]*\}", text)
    candidate = m.group(0) if m else text
    try:
        obj = json.loads(candidate)
        idx = int(obj.get("action_idx", -1))
        if 0 <= idx < n_actions:
            return idx
    except Exception:
        pass
    # fallback regex: 提取数字
    nums = re.findall(r"\d+", text)
    for n in nums:
        i = int(n)
        if 0 <= i < n_actions:
            return i
    return None


def llm_pick_action(
    client: OpenAI,
    current_sid: str,
    entities: List[str],
    actions: List[str],
    neighbors: List[int],
    recent_states: List[str],
    memory_hints: List[Dict[str, Any]],
    accountant: CostAccountant,
    max_completion_tokens: int = 60,
    retries: int = 2,
) -> Tuple[int, bool]:
    """
    Return (action_idx, is_llm_picked). fallback=uniform random over n actions if LLM fails.
    """
    # 构造精简 prompt (省 tokens)
    n_actions = len(actions)
    # 关联 action -> neighbor (对齐 v2 pipeline: action idx 映射 to neighbor idx via rng in walker)
    # 我们让 LLM 选一个 action 索引; walker 之后按 rng 从 neighbors 里挑 (与 random-walk 语义一致)
    hints_text = ""
    if memory_hints:
        hh = [f"{h['memory_id'].split('::')[-1]} (freq={h['access_freq_running']})"
              for h in memory_hints[:3]]
        hints_text = "\n  memory_hints: " + ", ".join(hh)
    recent_text = ", ".join(recent_states[-5:]) if recent_states else "(none)"
    system_msg = (
        "You are a walker choosing the next action in a text world. "
        "Reply with ONLY: {\"action_idx\": <int>} where int is 0..n_actions-1. "
        "No explanation."
    )
    user_msg = (
        f"current_state: {current_sid}\n"
        f"actions: {actions}\n"
        f"n_neighbors: {len(neighbors)}\n"
        f"recent_states: {recent_text}"
        f"{hints_text}\n"
        f"Choose one action index (0..{n_actions-1})."
    )

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": user_msg}],
                max_completion_tokens=max_completion_tokens,
            )
            if resp.usage:
                accountant.add(resp.usage.prompt_tokens or 0,
                                resp.usage.completion_tokens or 0)
            content = (resp.choices[0].message.content or "").strip()
            idx = _extract_action_idx(content, n_actions)
            if idx is not None:
                return idx, True
            last_err = ValueError(f"parse failed: {content[:80]!r}")
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))

    # fallback
    accountant.fallback_count += 1
    accountant.error_count += 1
    accountant.last_error = str(last_err)[:200] if last_err else None
    return random.Random().randrange(n_actions), False


# ==============================================================================
# 统计工具
# ==============================================================================


def gini(x: np.ndarray) -> float:
    if x.size == 0: return 0.0
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if x.sum() == 0: return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def summary_stats(freqs: List[int]) -> Dict[str, float]:
    arr = np.array(sorted(freqs, reverse=True), dtype=np.int64)
    if arr.size == 0:
        return {"n_unique": 0, "top1": 0, "median": 0.0, "max_over_median": 0.0,
                "skew": 0.0, "gini": 0.0, "top10pct_share": 0.0, "singleton_fraction": 0.0,
                "mean": 0.0, "std": 0.0, "total": 0}
    total = int(arr.sum())
    med = float(np.median(arr))
    top1 = int(arr[0])
    mx_med = (float(arr[0]) / med) if med > 0 else float("inf")
    skew = float(spstats.skew(arr)) if arr.size > 1 else 0.0
    g = float(gini(arr))
    top10n = max(1, int(np.ceil(arr.size * 0.1)))
    top10_share = float(arr[:top10n].sum()) / max(1, total)
    singletons = float((arr == 1).sum()) / arr.size
    return {"n_unique": int(arr.size), "top1": top1, "median": med, "max_over_median": mx_med,
            "skew": skew, "gini": g, "top10pct_share": top10_share,
            "singleton_fraction": singletons, "mean": float(arr.mean()),
            "std": float(arr.std()), "total": total}


def fit_lognormal_ks(freqs: List[int]) -> Dict[str, float]:
    arr = np.array(freqs, dtype=np.float64)
    arr = arr[arr > 0]
    if arr.size < 10:
        return {"mu": 0.0, "sigma": 0.0, "ks_from_lognormal": 0.0}
    log_arr = np.log(arr)
    mu = float(log_arr.mean())
    sigma = float(log_arr.std(ddof=1))
    if sigma > 0:
        cdf_th = spstats.norm.cdf((np.sort(log_arr) - mu) / sigma)
        cdf_em = np.arange(1, arr.size + 1) / arr.size
        ks = float(np.max(np.abs(cdf_th - cdf_em)))
    else:
        ks = 0.0
    return {"mu": mu, "sigma": sigma, "ks_from_lognormal": ks}


def pilot_pl_lrt(freqs: List[int], x_min: int = 1) -> Dict[str, Any]:
    """
    简化 Clauset PL vs lognormal LRT (pilot 用, n_boot=100).
    正式 n_boot=200 让 data_scientist 做, 这里给 quick indication.

    返回 (log_lik_pl, log_lik_lognormal, LR, p_two_sided).
    使用 Clauset et al. Section 4/5 的 alpha_hat MLE for discrete/continuous PL.
    """
    arr = np.array(freqs, dtype=np.float64)
    arr = arr[arr >= x_min]
    n = arr.size
    if n < 20:
        return {"n": n, "note": "n<20, skip LRT"}

    # Continuous MLE for PL: alpha = 1 + n * (sum log(x_i / x_min))^{-1}
    log_ratios = np.log(arr / x_min)
    alpha = 1.0 + n / np.sum(log_ratios)
    if alpha <= 1.0:
        return {"n": n, "note": f"alpha_hat={alpha:.3f} invalid"}

    # LL_PL (continuous): sum log[(alpha-1)/x_min * (x_i/x_min)^{-alpha}]
    ll_pl = n * np.log((alpha - 1.0) / x_min) - alpha * np.sum(log_ratios)

    # LL_lognormal: MLE on log(arr)
    log_arr = np.log(arr)
    mu = log_arr.mean()
    sigma = log_arr.std(ddof=1)
    if sigma <= 0:
        return {"n": n, "alpha_hat": alpha, "ll_pl": ll_pl, "note": "sigma=0"}
    # continuous truncated at x_min lognormal (approx, 忽略截断修正, pilot 足够)
    ll_ln = np.sum(spstats.lognorm.logpdf(arr, s=sigma, scale=np.exp(mu)))

    # Vuong-style normalized LR (Clauset eq 8-9)
    # per-point LL difference variance
    ll_pl_pts = np.log((alpha - 1.0) / x_min) - alpha * log_ratios
    ll_ln_pts = spstats.lognorm.logpdf(arr, s=sigma, scale=np.exp(mu))
    diff = ll_pl_pts - ll_ln_pts
    LR = float(np.sum(diff))
    sd = float(np.std(diff, ddof=1))
    if sd == 0:
        return {"n": n, "alpha_hat": float(alpha), "ll_pl": float(ll_pl),
                "ll_lognormal": float(ll_ln), "LR": LR, "note": "sd=0"}
    Z = LR / (sd * np.sqrt(n))
    p_two = float(2 * (1 - spstats.norm.cdf(abs(Z))))
    return {
        "n": int(n), "alpha_hat": float(alpha),
        "ll_pl": float(ll_pl), "ll_lognormal": float(ll_ln),
        "LR": LR, "Z": float(Z), "p_two_sided": p_two,
        "interpretation": "PL better" if LR > 0 else "lognormal better",
    }


# ==============================================================================
# 主 run
# ==============================================================================


def run_llm_policy(
    n_nodes: int, n_steps: int, seed: int,
    reservoir_capacity: int, top_k_retrieve: int,
    graph_type: str, out_dir: str, budget_usd: float,
    checkpoint_every: int = 500,
) -> Dict[str, Any]:
    t0 = time.time()
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    tag = f"{graph_type}_v{n_nodes}_s{seed}_llm"
    print(f"[SETUP] {graph_type} v{n_nodes} seed={seed} target_N={n_steps} budget=${budget_usd:.2f}", flush=True)

    g = build_graph(graph_type, n_nodes, seed=seed)
    payloads = build_state_payloads(g, seed=seed)
    print(f"[SETUP] graph edges={g.number_of_edges()}", flush=True)

    rng_walk = random.Random(seed + 100)
    rng_mem = random.Random(seed + 200)
    rng_fallback = random.Random(seed + 300)

    neighbors_cache = {n: list(g.successors(n)) for n in g.nodes()}
    action_cache = {n: payloads[n].actions for n in g.nodes()}

    client = make_client()
    accountant = CostAccountant(budget_usd=budget_usd)

    mem = StateAwareReservoirMemory(capacity=reservoir_capacity, rng=rng_mem)

    state_counter: Counter = Counter()
    trans_counter: Counter = Counter()
    half = n_steps // 2
    state_first: Counter = Counter()
    state_second: Counter = Counter()
    mem_first: Counter = Counter()
    mem_second: Counter = Counter()

    mem_time_records: List[Tuple[int, str, str, int]] = []
    action_records: List[Dict[str, Any]] = []  # per-step audit
    recent_states: List[str] = []

    current = rng_walk.choice(list(g.nodes()))
    actual_steps = 0
    stopped_reason = "completed"

    for step in range(n_steps):
        sid = f"v_{current:04d}"
        state_counter[sid] += 1
        if step < half:
            state_first[sid] += 1
        else:
            state_second[sid] += 1
        recent_states.append(sid)

        if step + 1 >= n_steps:
            actual_steps = step + 1
            break

        # === LLM policy pick action ===
        neighbors = neighbors_cache[current]
        actions = action_cache[current]
        n_acts = len(actions)

        # retrieve memory hints first (与 v2 cleanroom 一致的 retrieve 位置)
        hints = mem.retrieve(current_state=sid, k=top_k_retrieve, step=step)

        # LLM 选 action idx
        picked_idx, is_llm = llm_pick_action(
            client, sid, payloads[current].entities, actions,
            neighbors, recent_states, hints, accountant,
        )
        # Convert action idx to next_node (与 v2 一致: walker 用 rng 均匀选 neighbor).
        # 但 policy 主动的 action idx 应影响 neighbor 选择 → 我们让 action idx 决定 neighbor sub-index.
        # 具体: 把 neighbors 均分成 n_acts 段, action_idx 段内再 uniform (fallback 语义)。
        # 这样保持 walker 的确定性来自 policy 决策而非 pure rng, 与 v2 语义有变化, 但为了让 LLM policy 真的影响轨迹, 必要。
        if len(neighbors) == 0:
            nxt_node = current
        else:
            seg_size = max(1, len(neighbors) // n_acts)
            seg_start = picked_idx * seg_size
            seg_end = seg_start + seg_size if picked_idx < n_acts - 1 else len(neighbors)
            seg = neighbors[seg_start:seg_end] if seg_start < len(neighbors) else neighbors
            nxt_node = rng_walk.choice(seg) if seg else rng_walk.choice(neighbors)
        action = actions[picked_idx]
        nxt_sid = f"v_{nxt_node:04d}"
        tid = f"{sid}::{action}::{nxt_sid}"

        # memory write
        mid = f"tx_{tid}"
        content = f"transition {sid}--{action}-->{nxt_sid}"
        for ev in mem.write(mid, content, prev=sid, action=action, nxt=nxt_sid, step=step):
            if step < half:
                mem_first[ev["memory_id"]] += 1
            else:
                mem_second[ev["memory_id"]] += 1
            mem_time_records.append((step, ev["memory_id"], ev["access_kind"], -1))

        # 补记 hints 的 access counts
        for ev in hints:
            if step < half:
                mem_first[ev["memory_id"]] += 1
            else:
                mem_second[ev["memory_id"]] += 1
            mem_time_records.append((step, ev["memory_id"], ev["access_kind"], ev["retrieval_rank"]))

        trans_counter[tid] += 1
        action_records.append({
            "step": step, "prev": sid, "action_idx": picked_idx, "action": action,
            "next": nxt_sid, "is_llm_picked": is_llm,
        })

        current = nxt_node
        actual_steps = step + 1

        # 预算检查 (每 100 步)
        if step % 100 == 99:
            used = accountant.estimated_usd()
            if step % 500 == 499:
                print(f"  step {step+1}: used=${used:.3f} / ${budget_usd:.2f} "
                      f"({accountant.api_calls} calls, {accountant.fallback_count} fallbacks)", flush=True)
            if accountant.budget_exceeded():
                stopped_reason = "budget_exceeded"
                print(f"  [BUDGET] hit ${used:.3f} >= ${budget_usd:.2f}, stopping at step {step+1}", flush=True)
                break

    # 落盘
    state_freqs = list(state_counter.values())
    trans_freqs = list(trans_counter.values())
    mem_freqs = list(mem.access_counter.values())

    with open(os.path.join(results_dir, f"{tag}_state_counts.json"), "w") as f:
        json.dump(dict(state_counter), f)
    with open(os.path.join(results_dir, f"{tag}_trans_counts.json"), "w") as f:
        json.dump(dict(trans_counter), f)
    with open(os.path.join(results_dir, f"{tag}_mem_counts.json"), "w") as f:
        json.dump(dict(mem.access_counter), f)

    with open(os.path.join(results_dir, f"{tag}_state_first_half.json"), "w") as f:
        json.dump(dict(state_first), f)
    with open(os.path.join(results_dir, f"{tag}_state_second_half.json"), "w") as f:
        json.dump(dict(state_second), f)
    with open(os.path.join(results_dir, f"{tag}_mem_first_half.json"), "w") as f:
        json.dump(dict(mem_first), f)
    with open(os.path.join(results_dir, f"{tag}_mem_second_half.json"), "w") as f:
        json.dump(dict(mem_second), f)

    with open(os.path.join(results_dir, f"{tag}_mem_time.jsonl"), "w") as f:
        for (st, mid, kind, rank) in mem_time_records:
            f.write(json.dumps({"step": st, "mid": mid, "kind": kind, "rank": rank}) + "\n")

    with open(os.path.join(results_dir, f"{tag}_actions.jsonl"), "w") as f:
        for r in action_records:
            f.write(json.dumps(r) + "\n")

    st = summary_stats(state_freqs)
    tr = summary_stats(trans_freqs)
    me = summary_stats(mem_freqs)
    st_first = summary_stats(list(state_first.values()))
    st_second = summary_stats(list(state_second.values()))
    me_first = summary_stats(list(mem_first.values()))
    me_second = summary_stats(list(mem_second.values()))
    me_shape = fit_lognormal_ks(mem_freqs)
    me_lrt = pilot_pl_lrt(mem_freqs)

    # 决策多样性: action idx 相邻步的相关性 + action 序列的熵
    if action_records:
        idx_seq = [r["action_idx"] for r in action_records]
        # entropy
        c = Counter(idx_seq)
        p = np.array(list(c.values())) / len(idx_seq)
        entropy = float(-np.sum(p * np.log2(p + 1e-12)))
        # 与前一步 idx 相同的比例
        same_prev = float(sum(1 for i in range(1, len(idx_seq)) if idx_seq[i] == idx_seq[i-1]) / max(1, len(idx_seq) - 1))
        # llm-picked 比例
        llm_pick_rate = float(sum(1 for r in action_records if r["is_llm_picked"]) / len(action_records))
    else:
        entropy = 0.0; same_prev = 0.0; llm_pick_rate = 0.0

    meta = {
        "hypothesis": "H0.llm_policy_pilot",
        "graph_type": graph_type, "n_nodes": n_nodes, "seed": seed,
        "target_n_steps": n_steps, "actual_steps": actual_steps,
        "stopped_reason": stopped_reason,
        "reservoir_capacity": reservoir_capacity, "top_k_retrieve": top_k_retrieve,
        "elapsed_sec": time.time() - t0,
        "policy": {
            "type": "gpt-5.4-mini_llm_policy",
            "model": LLM_MODEL,
            "endpoint": LLM_API_BASE_URL,
            "llm_pick_rate": llm_pick_rate,
            "fallback_count": accountant.fallback_count,
            "action_entropy_bits": entropy,
            "same_as_prev_step_rate": same_prev,
            "error_count": accountant.error_count,
            "last_error": accountant.last_error,
        },
        "cost": {
            "budget_usd": budget_usd,
            "tokens_prompt": accountant.tokens_prompt,
            "tokens_completion": accountant.tokens_completion,
            "api_calls": accountant.api_calls,
            "estimated_usd": accountant.estimated_usd(),
            "price_prompt_per_1M": PRICE_PROMPT_PER_1M,
            "price_completion_per_1M": PRICE_COMPL_PER_1M,
        },
        "state_stats": st, "trans_stats": tr, "mem_stats": me,
        "mem_shape_lognormal": me_shape,
        "mem_pl_vs_lognormal_lrt_pilot": me_lrt,
        "temporal_stability": {
            "state_first_half": st_first, "state_second_half": st_second,
            "mem_first_half": me_first, "mem_second_half": me_second,
        },
        "memory_bookkeeping": {
            "n_write_events": mem.write_events,
            "n_read_events": mem.read_events,
            "n_evict_events": mem.evict_events,
            "n_conflicts": mem.conflicts,
            "n_slots_used": len(mem.slots),
        },
    }
    with open(os.path.join(results_dir, f"{tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[DONE] actual_steps={actual_steps} elapsed={time.time()-t0:.1f}s cost=${accountant.estimated_usd():.3f}", flush=True)
    print(f"       mem gini={me['gini']:.3f} skew={me['skew']:.2f} max/med={me['max_over_median']:.1f}", flush=True)
    print(f"       LRT pilot: {me_lrt.get('interpretation','n/a')} (LR={me_lrt.get('LR',0):.1f}, p={me_lrt.get('p_two_sided',1):.3f}, alpha_hat={me_lrt.get('alpha_hat',0):.2f})", flush=True)
    print(f"       policy: llm_pick={llm_pick_rate*100:.1f}%, entropy={entropy:.2f}bit, same_prev={same_prev*100:.1f}%", flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--n_nodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graph_type", type=str, default="scale_free")
    parser.add_argument("--reservoir_capacity", type=int, default=200)
    parser.add_argument("--top_k_retrieve", type=int, default=3)
    parser.add_argument("--budget_usd", type=float, default=5.0)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    run_llm_policy(
        n_nodes=args.n_nodes, n_steps=args.n_steps, seed=args.seed,
        reservoir_capacity=args.reservoir_capacity, top_k_retrieve=args.top_k_retrieve,
        graph_type=args.graph_type, out_dir=args.out_dir, budget_usd=args.budget_usd,
    )


if __name__ == "__main__":
    main()
