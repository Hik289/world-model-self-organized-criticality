from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import powerlaw
from scipy import signal, stats

from worldmodelsoc.memory.reservoir import summary_stats


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
