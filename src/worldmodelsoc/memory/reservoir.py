from __future__ import annotations

import contextlib
import io
import math
import random
import warnings
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import powerlaw
from scipy import signal, stats
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

    def retrieve(self, current_state: str, step: int = 0) -> List[Dict[str, Any]]:
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


FAMILIES = ("power_law", "truncated_power_law", "lognormal", "exponential")


def finite(value):
    number = float(value)
    return number if np.isfinite(number) else None


def fit_counts(values: np.ndarray, xmin: float | None = None):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", RuntimeWarning)
        return powerlaw.Fit(values, xmin=xmin, discrete=True, estimate_discrete=False, verbose=False)


def bootstrap_gof(values: np.ndarray, fitted, family: str, samples: int, seed: int, fixed_xmin: float | None) -> dict:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    distribution = getattr(fitted, family)
    observed = float(distribution.D)
    if not np.isfinite(observed):
        raise ValueError(f"non-finite observed KS statistic for {family}")
    body = values[values < fitted.xmin]
    probability = np.mean(values >= fitted.xmin)
    rng = np.random.default_rng(seed)
    distances = []
    failures = 0
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        for _ in range(samples):
            tail_size = int(rng.binomial(len(values), probability))
            generated = np.asarray(distribution.generate_random(tail_size, estimate_discrete=False)) if tail_size else np.empty(0)
            lower = rng.choice(body, len(values) - tail_size, replace=True) if len(values) > tail_size else np.empty(0)
            sample = np.concatenate((lower, generated))
            if len(np.unique(sample)) < 2 or not np.all(np.isfinite(sample)):
                failures += 1
                continue
            try:
                refitted = fit_counts(sample, fixed_xmin)
                distance = float(getattr(refitted, family).D)
            except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
                failures += 1
                continue
            if not np.isfinite(distance):
                failures += 1
                continue
            distances.append(distance)
    finally:
        np.random.set_state(state)
    valid = len(distances)
    p = (1 + sum(distance >= observed for distance in distances)) / (valid + 1) if valid and failures == 0 else None
    return {
        "observed_ks": observed, "requested_replicates": samples, "successful_replicates": valid,
        "failed_replicates": failures, "p_gof": p,
        "xmin_refitted": fixed_xmin is None,
    }


def audit_counts(counts, min_tail: int = 100, bootstrap: int = 1000, seed: int = 42, xmin: float | None = None) -> dict:
    values = np.asarray(list(counts), dtype=float)
    if min_tail < 2 or bootstrap < 1:
        raise ValueError("min_tail must be at least 2 and bootstrap must be positive")
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError("counts must be a one-dimensional sequence of finite non-negative integers")
    values = values[values > 0]
    result = {"n_positive": len(values), "min_tail": min_tail, "seed": seed, "discrete": True, "cutoff_selection": "fixed" if xmin is not None else "power_law_ks", "summary": summary_stats(values.astype(int).tolist())}
    top = np.sort(values)[-max(1, int(np.ceil(len(values) * 0.1))):]
    result["top10_log_count_sigma"] = float(np.std(np.log(top))) if len(top) else None
    result["top10_count_std"] = float(np.std(top)) if len(top) else None
    if len(values) < min_tail or len(np.unique(values)) < 2:
        return {**result, "status": "insufficient_counts", "eligible": False}
    fitted = fit_counts(values, xmin)
    n_tail = int(np.sum(values >= fitted.xmin))
    result.update(xmin=float(fitted.xmin), n_tail=n_tail, eligible=n_tail >= min_tail)
    if n_tail < min_tail:
        return {**result, "status": "insufficient_fitted_tail"}
    result["fits"] = {}
    for index, family in enumerate(FAMILIES):
        distribution = getattr(fitted, family)
        parameters = {name: finite(getattr(distribution, name)) for name in ("alpha", "Lambda", "mu", "sigma") if hasattr(distribution, name)}
        gof = bootstrap_gof(values, fitted, family, bootstrap, seed + index, xmin)
        result["fits"][family] = {
            "parameters": parameters, **gof,
            "compatible_at_0_10": gof["p_gof"] >= 0.1 if gof["p_gof"] is not None else None,
        }
    comparisons = []
    for first, second in (("power_law", "lognormal"), ("power_law", "exponential"), ("truncated_power_law", "power_law"), ("truncated_power_law", "lognormal"), ("truncated_power_law", "exponential")):
        nested = {first, second} == {"power_law", "truncated_power_law"}
        ratio, p_value = fitted.distribution_compare(first, second, nested=nested, normalized_ratio=not nested)
        comparisons.append({"first": first, "second": second, "loglikelihood_ratio": finite(ratio), "p_value": finite(p_value), "nested": nested, "normalized_ratio": not nested})
    result.update(status="audited", likelihood_comparisons=comparisons)
    return result


def psd_audit(series, sample_rate: float = 1.0, min_frequency: float | None = None, max_frequency: float | None = None) -> dict:
    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or sample_rate <= 0:
        raise ValueError("series must be finite and sample_rate must be positive")
    if len(values) < 16 or np.ptp(values) == 0:
        return {"status": "insufficient_temporal_variation", "n_samples": len(values)}
    frequency, density = signal.welch(values, fs=sample_rate, nperseg=min(256, len(values)), detrend="linear")
    keep = (frequency > 0) & (density > 0)
    if min_frequency is not None:
        keep &= frequency >= min_frequency
    if max_frequency is not None:
        keep &= frequency <= max_frequency
    if keep.sum() < 3:
        return {"status": "insufficient_frequency_bins", "n_samples": len(values)}
    fit = stats.linregress(np.log(frequency[keep]), np.log(density[keep]))
    return {
        "status": "fitted", "n_samples": len(values), "beta": float(-fit.slope),
        "r_squared": float(fit.rvalue ** 2), "slope_standard_error": float(fit.stderr),
        "frequency_range": [float(frequency[keep].min()), float(frequency[keep].max())],
        "frequencies": frequency[keep].tolist(), "spectral_density": density[keep].tolist(),
    }


class ControlledReservoir(StateAwareReservoirMemory):
    def __init__(self, capacity, rng, mode):
        super().__init__(capacity, rng)
        self.mode = mode

    def _evict_if_needed(self):
        if self.mode in ("uniform_frequency", "state_uniform"):
            if len(self.slots) >= self.capacity:
                del self.slots[self.rng.choice(sorted(self.slots))]
                self.evict_events += 1
        else:
            super()._evict_if_needed()

    def retrieve(self, current_state, k=3, step=0):
        if self.mode == "no_retrieval":
            return []
        candidates = list(self.slots.items())
        if self.mode != "uniform_frequency":
            local = [(key, value) for key, value in candidates if current_state in (value["prev"], value["next"])]
            candidates = local or candidates
        if self.mode == "state_uniform":
            candidates = self.rng.sample(candidates, min(k, len(candidates)))
        else:
            candidates.sort(key=lambda item: (-item[1]["freq"], item[0]))
            candidates = candidates[:k]
        events = []
        for rank, (key, value) in enumerate(candidates, 1):
            self.access_counter[key] += 1
            self.read_events += 1
            events.append({"memory_id": key, "rank": rank, "content": value["content"], "step": step})
        return events
