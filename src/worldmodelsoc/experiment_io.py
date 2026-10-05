from __future__ import annotations

import json
import importlib.metadata
import platform
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from worldmodelsoc.llm_config import get_llm_model, make_openai_client


def read_records(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty input: {path}")
    records = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("expected JSON array or JSONL objects")
    return records


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def provenance(config: dict) -> dict:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    versions = {}
    for package in ("numpy", "scipy", "networkx", "powerlaw", "openai", "tiktoken", "alfworld"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"config": config, "git_commit": result.stdout.strip() or None, "python": platform.python_version(), "packages": versions}


class ChatRecorder:
    def __init__(self, path: str | Path, model: str | None = None, max_calls: int = 10000):
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.path = Path(path)
        self.model = model or get_llm_model()
        self.max_calls = max_calls
        self.client = None
        self.usage = Counter()
        self.calls = []

    def ask(self, messages: list[dict], phase: str, max_tokens: int = 256, model: str | None = None) -> str:
        if len(self.calls) >= self.max_calls:
            raise RuntimeError("maximum API call count reached")
        if self.client is None:
            self.client = make_openai_client()
        response = self.client.chat.completions.create(
            model=model or self.model, messages=messages, temperature=0,
            max_completion_tokens=max_tokens,
        )
        if response.usage is None:
            raise RuntimeError("API did not return usage; actual prompt-token accounting is unavailable")
        text = response.choices[0].message.content or ""
        usage = {"prompt_tokens": int(response.usage.prompt_tokens), "completion_tokens": int(response.usage.completion_tokens)}
        self.usage.update(usage)
        self.usage.update({f"{phase}_{key}": value for key, value in usage.items()})
        record = {"call_id": len(self.calls), "phase": phase, "model": model or self.model, "usage": usage, "response": text}
        self.calls.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return text


def prediction_metrics(records: list[dict], rare_states: set[str] | None = None) -> dict:
    visits = Counter(record["prev"] for record in records)
    if rare_states is None:
        count = max(1, len(visits) // 2) if visits else 0
        rare_states = set(sorted(visits, key=lambda state: (visits[state], state))[:count])
    tail = [record for record in records if record["prev"] in rare_states]
    error = lambda row: row.get("prediction") != row["next"]
    return {
        "n_steps": len(records), "state_counts": dict(visits), "rare_states": sorted(rare_states),
        "prediction_error": sum(map(error, records)) / len(records) if records else None,
        "tail_prediction_error": sum(map(error, tail)) / len(tail) if tail else None,
        "tail_visits": len(tail), "tail_errors": sum(map(error, tail)),
        "invalid_predictions": sum(record.get("prediction") is None for record in records),
    }


def paired_summary(rows: list[dict], baseline: str, method: str, metrics: list[str], seed: int = 42, resamples: int = 2000) -> dict:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    grouped = {}
    for row in rows:
        key = row["pair_id"]
        name = row["method"]
        if name in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate pair/method: {key}/{name}")
        grouped[key][name] = row
    pairs = [values for values in grouped.values() if baseline in values and method in values]
    rng = np.random.default_rng(seed)
    result = {"baseline": baseline, "method": method, "n_pairs": len(pairs), "n_unpaired": len(grouped) - len(pairs), "metrics": {}}
    for metric in metrics:
        valid = [pair for pair in pairs if pair[baseline].get(metric) is not None and pair[method].get(metric) is not None]
        if not valid:
            continue
        before = np.array([pair[baseline][metric] for pair in valid], dtype=float)
        after = np.array([pair[method][metric] for pair in valid], dtype=float)
        if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
            raise ValueError(f"non-finite metric: {metric}")
        differences = after - before
        draws = rng.choice(differences, size=(resamples, len(valid)), replace=True).mean(axis=1)
        mean_before = float(before.mean())
        result["metrics"][metric] = {
            "n_pairs": len(valid), "baseline_mean": mean_before, "method_mean": float(after.mean()),
            "paired_difference": float(differences.mean()),
            "paired_difference_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "reduction_percent": 100 * (1 - float(after.mean()) / mean_before) if mean_before else None,
        }
    return result
