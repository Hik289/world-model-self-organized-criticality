"""
Seven-module API pipeline used by the toy sanity check.

Modules:
  1. State Extractor         — 从 observation 抽 canonical state_id
  2. Transition Extractor    — 从 (prev_state, action, obs) 抽 canonical transition
  3. Memory Writer           — 把 state / transition 写入 KV memory
  4. Memory Retriever        — 从 memory 按 top-K 检索
  5. Next-State Predictor    — 给出下一状态预测
  6. Prediction Evaluator    — 打 error_magnitude 分
  7. Token Profiler          — 每 100 step (或每 5 step 在 toy sanity) snapshot 一次 token 使用

LLM calls use an OpenAI-compatible chat-completions endpoint.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from worldmodelsoc.llm_config import LLM_MODEL, make_openai_client


# ==============================================================================
# LLM client
# ==============================================================================

def make_client(key_path: str | None = None) -> OpenAI:
    return make_openai_client(key_path)


# ==============================================================================
# 全局 token 计数器
# ==============================================================================

@dataclass
class TokenAccumulator:
    tokens_prompt: int = 0
    tokens_completion: int = 0
    api_calls: int = 0


def _chat(client: OpenAI, system: str, user: str, acc: TokenAccumulator,
          max_completion_tokens: int = 200, retries: int = 3) -> str:
    """
    单次 chat completion, 累积 token 到 acc。
    """
    if max_completion_tokens < 1:
        raise ValueError("max_completion_tokens must be at least 1")
    if retries < 1:
        raise ValueError("retries must be at least 1")
    last_err: Exception | None = None
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=max_completion_tokens,
            )
            if resp.usage:
                acc.tokens_prompt += resp.usage.prompt_tokens or 0
                acc.tokens_completion += resp.usage.completion_tokens or 0
            acc.api_calls += 1
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    从 LLM 输出里抽第一个合法 JSON object。宽容处理 code fence 等噪声。
    失败时返回空 dict。
    """
    if not text:
        return {}
    # 剥 code fence
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # 尝试逐段修剪
    for end in range(len(candidate), 0, -1):
        try:
            return json.loads(candidate[:end])
        except json.JSONDecodeError:
            continue
    return {}


# ==============================================================================
# Module 1: State Extractor
# ==============================================================================

def state_extractor(client: OpenAI, observation: str,
                    canonical_state_ids: List[str],
                    acc: TokenAccumulator) -> str:
    """
    从 observation 文本抽 canonical state_id。返回单一 state_id 字符串。
    """
    system = (
        "You are a State Extractor for a text-based world model. "
        "You must map a natural-language observation to ONE canonical state id from the given list. "
        "Reply ONLY with a JSON object of the form: {\"state_id\": \"<one of the canonical ids>\"}. "
        "No explanation, no code fences."
    )
    user = (
        f"Canonical state ids: {canonical_state_ids}\n\n"
        f"Observation:\n{observation}\n\n"
        "Return only the JSON."
    )
    raw = _chat(client, system, user, acc, max_completion_tokens=1500)
    obj = _extract_json_object(raw)
    sid_value = obj.get("state_id", "")
    sid = sid_value.strip() if isinstance(sid_value, str) else ""
    # fallback: 简单关键词匹配
    if sid not in canonical_state_ids:
        lower_obs = observation.lower()
        for cid in canonical_state_ids:
            if cid.replace("_", " ") in lower_obs or cid in lower_obs:
                sid = cid
                break
        else:
            sid = "unknown"
    return sid


# ==============================================================================
# Module 2: Transition Extractor
# ==============================================================================

def transition_extractor(client: OpenAI, prev_state: str, action: str, next_state: str,
                         acc: TokenAccumulator) -> str:
    """
    构造 canonical transition_id。LLM 只做 sanity 校验 (确认这个 transition 语义合理),
    canonical_id 由代码拼接 (确保稳定)。
    """
    system = (
        "You are a Transition Extractor. Given (prev_state, action, next_state), "
        "reply with a JSON object: {\"transition_id\": \"<prev>::<action>::<next>\", \"plausible\": true|false}. "
        "Set plausible=true if the transition looks physically/logically consistent with a house-navigation setup."
    )
    user = f"prev_state={prev_state}, action={action}, next_state={next_state}"
    _chat(client, system, user, acc, max_completion_tokens=1500)
    return f"{prev_state}::{action}::{next_state}"


# ==============================================================================
# Module 3 + 4: Memory Writer / Retriever  (KV, 支持 top-K 检索)
# ==============================================================================

@dataclass
class MemoryStore:
    """
    简单 KV memory: key=memory_id -> {content, freq, first_seen_step, last_seen_step}.
    Retriever 用简单 keyword overlap 打分, 返回 top-K。
    """
    entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    write_events: int = 0
    read_events: int = 0
    conflicts: int = 0  # 主键冲突计数 (应该始终 = 0, invariant B3)

    def write(self, memory_id: str, content: str, step: int) -> Dict[str, Any]:
        if memory_id in self.entries:
            # overwrite 语义: 更新计数, 不冲突 (冲突指要写新 key 但 hash 撞了旧 key 的不同 content)
            rec = self.entries[memory_id]
            if rec["content"] != content:
                # 内容变了但用同一 key: 计入 conflict (invariant test)
                self.conflicts += 1
            rec["freq"] = rec.get("freq", 0) + 1
            rec["last_seen_step"] = step
            self.write_events += 1
            return {"memory_id": memory_id, "access_kind": "overwrite", "freq": rec["freq"]}
        rec = {
            "memory_id": memory_id,
            "content": content,
            "freq": 1,
            "first_seen_step": step,
            "last_seen_step": step,
        }
        self.entries[memory_id] = rec
        self.write_events += 1
        return {"memory_id": memory_id, "access_kind": "write", "freq": 1}

    def _score(self, query: str, content: str) -> float:
        q = set(re.findall(r"\w+", query.lower()))
        c = set(re.findall(r"\w+", content.lower()))
        if not q or not c:
            return 0.0
        return len(q & c) / max(1, len(q))

    def retrieve(self, query: str, top_k: int = 3, step: int = 0) -> List[Dict[str, Any]]:
        scored = []
        for mid, rec in self.entries.items():
            s = self._score(query, rec["content"])
            scored.append((s, mid, rec))
        scored.sort(key=lambda x: -x[0])
        top = scored[:top_k]
        out = []
        for rank, (score, mid, rec) in enumerate(top):
            rec["freq"] = rec.get("freq", 0)  # keep as-is; read does not bump content freq
            # 但 access_freq_running 是 access counter, 我们分开维护
            rec_access = rec.setdefault("_access_count", 0)
            rec["_access_count"] = rec_access + 1
            self.read_events += 1
            out.append({
                "memory_id": mid,
                "access_kind": "read",
                "retrieval_rank": rank,
                "retrieval_score": float(score),
                "access_freq_running": rec["_access_count"],
                "memory_tokens": max(1, len(rec["content"]) // 4),
            })
        return out


# ==============================================================================
# Module 5: Next-State Predictor
# ==============================================================================

def next_state_predictor(client: OpenAI, prev_state: str, action: str,
                         canonical_state_ids: List[str],
                         memory_hints: List[Dict[str, Any]],
                         acc: TokenAccumulator) -> Tuple[str, float]:
    """
    预测下一 state。返回 (predicted_state_id, confidence in [0,1])。
    memory_hints: retriever 返回的相关记忆列表 (可以含 transition 信息)。
    """
    hints_text = "\n".join(
        f"- {h.get('memory_id','')} (rank={h.get('retrieval_rank','?')}, score={h.get('retrieval_score','?')})"
        for h in memory_hints
    )
    system = (
        "You are a Next-State Predictor for a text world model. "
        "Given the previous state, the action taken, and retrieval hints from memory, "
        "predict the most likely next canonical state id. "
        "Reply ONLY with JSON: {\"predicted_state_id\": \"<id>\", \"confidence\": <0..1>}."
    )
    user = (
        f"Canonical state ids: {canonical_state_ids}\n"
        f"prev_state: {prev_state}\n"
        f"action: {action}\n"
        f"memory hints:\n{hints_text if hints_text else '(none)'}\n\n"
        "Return only the JSON."
    )
    raw = _chat(client, system, user, acc, max_completion_tokens=1500)
    obj = _extract_json_object(raw)
    pred_value = obj.get("predicted_state_id", "")
    pred = pred_value if isinstance(pred_value, str) else ""
    try:
        conf = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    if not math.isfinite(conf):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    if pred not in canonical_state_ids:
        pred = "unknown"
    return pred, conf


# ==============================================================================
# Module 6: Prediction Evaluator
# ==============================================================================

def prediction_evaluator(predicted: str, actual: str,
                         canonical_state_ids: List[str],
                         adjacency_lookup: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    评估 prediction 正误。error_magnitude ∈ [0,1]:
      0 = 完全正确
      0.5 = 预测在 actual 的邻居里 (partial credit)
      1 = 完全错
    """
    if predicted == actual:
        return {"prediction_correct": True, "error_magnitude": 0.0}
    if predicted == "unknown" or actual == "unknown":
        return {"prediction_correct": False, "error_magnitude": 1.0}
    neigh = adjacency_lookup.get(actual, [])
    if predicted in neigh:
        return {"prediction_correct": False, "error_magnitude": 0.5}
    return {"prediction_correct": False, "error_magnitude": 1.0}


# ==============================================================================
# Module 7: Token Profiler
# ==============================================================================

def token_profiler_snapshot(acc: TokenAccumulator, memory_token_estimate: int) -> Dict[str, int]:
    return {
        "tokens_prompt": acc.tokens_prompt,
        "tokens_completion": acc.tokens_completion,
        "tokens_memory_in_context": memory_token_estimate,
        "api_calls": acc.api_calls,
        "budget_remaining": None,
    }
