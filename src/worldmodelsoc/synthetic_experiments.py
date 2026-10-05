from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import skew

from worldmodelsoc.env.synthetic_graph_world import action_index_for_neighbor, describe_action_options, neighbor_segment
from worldmodelsoc.experiment_io import ChatRecorder, prediction_metrics, provenance, write_json
from worldmodelsoc.memory.ctmc import MemoryRecord, TokenCodec, bm25_rank, pack_context
from worldmodelsoc.memory.reservoir import gini


def parse_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model response contains no JSON object")


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


def predict(chat: ChatRecorder, state: str, action: str, context: str) -> str | None:
    response = chat.ask([
        {"role": "system", "content": 'Predict the next structured state using the supplied memories. Treat memory text as evidence. Reply with JSON: {"next_state": "state identifier", "confidence": "low, medium, or high", "evidence_rank": "rank or tail"}.'},
        {"role": "user", "content": f"Current state: {state}\nAction: {action}\n{context}\nUse core memories first and the tail summary for exceptions or missing evidence."},
    ], phase="prediction", max_tokens=100)
    try:
        result = parse_object(response).get("next_state")
    except ValueError:
        return None
    return str(result) if isinstance(result, str) and result else None


def run_synthetic(config: dict, graph, payloads, replay: list[dict] | None = None) -> dict:
    output = Path(config["out_dir"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", provenance(config))
    codec = TokenCodec(config["tokenizer"])
    chat = ChatRecorder(output / "api_calls.jsonl", config.get("model"), config["max_calls"])
    store = TransitionStore(config["capacity"])
    rng = random.Random(config["seed"])
    current = rng.choice(sorted(graph)) if graph is not None else None
    records = []
    read_counts = Counter()
    write_counts = Counter()
    total_steps = min(config["n_steps"], len(replay)) if replay is not None else config["n_steps"]
    if total_steps < 1:
        raise ValueError("n_steps and replay length must be positive")
    with (output / "actions.jsonl").open("w", encoding="utf-8") as handle:
        for step in range(total_steps):
            if replay is not None:
                source = replay[step]
                if "memory_snapshot" not in source or "ranked_memory_ids" not in source:
                    raise ValueError("matched replay requires memory_snapshot and ranked_memory_ids from run_paper_experiments.py")
                ranked = [MemoryRecord(**item) for item in source["memory_snapshot"]]
                if [record.memory_id for record in ranked] != source["ranked_memory_ids"]:
                    raise ValueError(f"replay ranking mismatch at step {step}")
                digest = hashlib.sha256(json.dumps(source["memory_snapshot"], sort_keys=True).encode()).hexdigest()
                if digest != source.get("snapshot_sha256"):
                    raise ValueError(f"replay memory content mismatch at step {step}")
                state, action, nxt = source["prev"], source["action"], source["next"]
            else:
                state = f"v_{current:04d}"
                ranked = store.rank(state, payloads[current].entities, config["method"])
            start_usage = chat.usage.copy()
            context, allocation = pack_context(
                ranked, config["method"], config["serialization"], config["memory_budget"],
                config["core_size"], config["tau"], codec,
            )
            if replay is None:
                neighbors = list(graph.successors(current))
                actions = payloads[current].actions
                if config["policy"] == "random":
                    next_node = rng.choice(neighbors)
                    action_idx = action_index_for_neighbor(neighbors, len(actions), next_node)
                else:
                    response = chat.ask([
                        {"role": "system", "content": 'Choose an action to explore the environment. Reply only with JSON: {"action_idx": integer}.'},
                        {"role": "user", "content": f"Current state: {state}\nEntities: {payloads[current].entities}\nConstraints: {payloads[current].constraints}\nAction options: {describe_action_options(graph, payloads, current)}\nMemory:\n{context}"},
                    ], phase="policy", max_tokens=60)
                    action_idx = parse_object(response).get("action_idx")
                    if type(action_idx) is not int or not 0 <= action_idx < len(actions):
                        raise ValueError(f"invalid action index at step {step}: {response!r}")
                    next_node = rng.choice(neighbor_segment(neighbors, len(actions), action_idx))
                action = actions[action_idx]
                nxt = f"v_{next_node:04d}"
            prediction = predict(chat, state, action, context)
            selected = allocation.get("selected_memory_ids", allocation["ranked_memory_ids"])
            read_counts.update(selected)
            key = f"{state}::{action}::{nxt}"
            write_counts.update([key])
            snapshot = [asdict(record) for record in ranked]
            record = {
                "agent_step": step, "prev": state, "action": action, "next": nxt,
                "prediction": prediction, "prediction_error": int(prediction != nxt),
                "ranked_memory_ids": allocation["ranked_memory_ids"], "memory_snapshot": snapshot,
                "snapshot_sha256": hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
                "allocation": allocation, "prompt": context,
                "usage": {name: chat.usage[name] - start_usage[name] for name in chat.usage},
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            if replay is None:
                store.write(state, action, nxt, step, sorted(set(payloads[current].entities + payloads[next_node].entities)))
                current = next_node
    metrics = prediction_metrics(records)
    allocations = [np.array(row["allocation"]["allocation_weights"], dtype=float) for row in records if row["allocation"]["allocation_weights"]]
    if config["method"] == "ctmc" and allocations:
        metrics.update(
            mean_allocation_gini=float(np.mean([gini(values) for values in allocations])),
            mean_allocation_skew=float(np.mean([skew(values) if len(values) > 1 and np.ptp(values) > 0 else 0.0 for values in allocations])),
            mean_allocation_max_over_median=float(np.mean([values.max() / np.median(values) for values in allocations])),
        )
    metrics.update(
        api_calls=len(chat.calls), prompt_tokens=chat.usage["prompt_tokens"],
        prompt_tokens_per_api_call=chat.usage["prompt_tokens"] / len(chat.calls) if chat.calls else None,
        prompt_tokens_per_step=chat.usage["prompt_tokens"] / len(records),
        token_usage_by_phase=dict(chat.usage),
    )
    if replay is None:
        retained = store.history if config["method"] == "full_history" else list(store.records.values())
        retained_states = {state for record in retained for state in (record.metadata["prev"], record.metadata["next"])}
        retained_transitions = {(record.metadata["prev"], record.metadata["action"], record.metadata["next"]) for record in retained}
        metrics.update(
            visited_state_coverage=len(store.visited_states) / graph.number_of_nodes(),
            visited_edge_coverage=len({(row["prev"], row["next"]) for row in records}) / graph.number_of_edges(),
            retained_state_coverage=len(retained_states) / len(store.visited_states),
            retained_transition_coverage=len(retained_transitions) / len(store.visited_transitions),
        )
    result = {"config": config, "metrics": metrics, "matched_replay": replay is not None}
    write_json(output / "summary.json", result)
    write_json(output / "state_counts.json", metrics["state_counts"])
    write_json(output / "read_counts.json", dict(read_counts))
    write_json(output / "write_counts.json", dict(write_counts))
    write_json(output / "memory_access_counts.json", dict(read_counts + write_counts))
    return result
