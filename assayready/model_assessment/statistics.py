from __future__ import annotations

import math
from statistics import NormalDist, mean, pstdev
from typing import Any, Iterable


def _finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def pearson(left: Iterable[Any], right: Iterable[Any]) -> float | None:
    pairs = []
    for first, second in zip(left, right):
        try:
            x, y = float(first), float(second)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    x_mean, y_mean = mean(xs), mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def spearman(left: Iterable[Any], right: Iterable[Any]) -> float | None:
    pairs = []
    for first, second in zip(left, right):
        try:
            x, y = float(first), float(second)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    return pearson(rankdata(list(xs)), rankdata(list(ys)))


def sample_size_for_two_sided_effect(
    *, effect_size: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Normal-approximation sample size for a paired standardized effect.

    This is a planning estimate, not an assay-specific power calculation.
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be positive.")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be between zero and one.")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha / 2.0)
    z_power = normal.inv_cdf(power)
    return max(2, math.ceil(((z_alpha + z_power) / effect_size) ** 2))


def replicate_noise(rows: list[dict[str, Any]], *, id_col: str, value_col: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        identifier = str(row.get(id_col) or "").strip()
        values = _finite([row.get(value_col)])
        if identifier and values:
            grouped.setdefault(identifier, []).extend(values)
    replicated = [values for values in grouped.values() if len(values) > 1]
    within_variances = [pstdev(values) ** 2 for values in replicated]
    all_values = [value for values in grouped.values() for value in values]
    within_variance = mean(within_variances) if within_variances else None
    total_variance = pstdev(all_values) ** 2 if len(all_values) > 1 else None
    reliability = None
    if within_variance is not None and total_variance and total_variance > 0:
        reliability = max(0.0, min(1.0, 1.0 - within_variance / total_variance))
    return {
        "unique_candidates": len(grouped),
        "replicated_candidates": len(replicated),
        "within_candidate_variance": within_variance,
        "total_variance": total_variance,
        "approximate_reliability": reliability,
        "interpretation": (
            "Approximate descriptive reliability; use a hierarchical assay-specific model for formal inference."
        ),
    }
