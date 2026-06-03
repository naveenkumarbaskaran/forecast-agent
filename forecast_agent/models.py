"""
Pure-Python time-series models.

No heavy dependencies beyond the standard library.
All functions operate on plain Python lists of floats.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence


def simple_moving_average(
    values: Sequence[float],
    periods: int,
    window: int | None = None,
) -> list[float]:
    """
    Forecast *periods* steps ahead using a simple moving average.

    The last ``window`` observations are averaged to produce a flat forecast
    for all future periods.

    Parameters
    ----------
    values:
        Historical time-series values (oldest first).
    periods:
        Number of future steps to forecast.
    window:
        Number of trailing observations to average.
        Defaults to ``min(len(values), 3)``.

    Returns
    -------
    list[float]
        Forecast values, length == ``periods``.
    """
    if not values:
        raise ValueError("values must not be empty")
    if periods < 1:
        raise ValueError("periods must be >= 1")

    n = len(values)
    w = window if window is not None else min(n, max(3, n // 4))
    w = max(1, min(w, n))

    avg = sum(values[-w:]) / w
    return [avg] * periods


def linear_trend(
    values: Sequence[float],
    periods: int,
) -> list[float]:
    """
    Forecast *periods* steps ahead by extrapolating an OLS linear trend.

    Fits ``y = a + b * t`` (t = 0, 1, ..., n-1) via the closed-form OLS
    solution, then evaluates at t = n, n+1, ..., n+periods-1.

    Parameters
    ----------
    values:
        Historical time-series values (oldest first).
    periods:
        Number of future steps to forecast.

    Returns
    -------
    list[float]
        Forecast values, length == ``periods``.
    """
    if len(values) < 2:
        raise ValueError("Need at least 2 data points for a linear trend.")
    if periods < 1:
        raise ValueError("periods must be >= 1")

    n = len(values)
    t = list(range(n))
    mean_t = (n - 1) / 2.0
    mean_y = sum(values) / n

    # slope b = S(t - mean_t)(y - mean_y) / S(t - mean_t)^2
    num = sum((t[i] - mean_t) * (values[i] - mean_y) for i in range(n))
    den = sum((ti - mean_t) ** 2 for ti in t)

    if den == 0:
        # All t identical — constant series.
        return [mean_y] * periods

    b = num / den
    a = mean_y - b * mean_t

    return [a + b * (n + i) for i in range(periods)]


def exponential_smoothing(
    values: Sequence[float],
    periods: int,
    alpha: float = 0.3,
) -> list[float]:
    """
    Forecast *periods* steps ahead via single (Holt) exponential smoothing.

    ``S_t = alpha * y_t + (1 - alpha) * S_{t-1}``

    The last smoothed level is used as a flat forecast for all future steps.

    Parameters
    ----------
    values:
        Historical time-series values (oldest first).
    periods:
        Number of future steps to forecast.
    alpha:
        Smoothing factor in (0, 1). Higher = more weight to recent data.

    Returns
    -------
    list[float]
        Forecast values, length == ``periods``.
    """
    if not values:
        raise ValueError("values must not be empty")
    if periods < 1:
        raise ValueError("periods must be >= 1")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")

    level = float(values[0])
    for y in values[1:]:
        level = alpha * y + (1 - alpha) * level

    return [level] * periods


def compute_confidence_interval(
    historical: Sequence[float],
    forecast: Sequence[float],
    z: float = 1.282,  # 80 % two-tailed
) -> tuple[list[float], list[float]]:
    """
    Compute symmetric confidence intervals around each forecast point.

    The interval half-width grows as ``z * std_dev * sqrt(h)`` where *h* is
    the horizon (1-indexed) and *std_dev* is the standard deviation of
    first-differences of the historical series.

    Parameters
    ----------
    historical:
        Historical observations.
    forecast:
        Forecast values (length == number of future periods).
    z:
        Z-score for the desired coverage (default 1.282 => 80 %).

    Returns
    -------
    tuple[list[float], list[float]]
        (lower_bounds, upper_bounds), each of length == len(forecast).
    """
    if len(historical) < 2:
        # No variance to estimate; return zero-width intervals.
        return (list(forecast), list(forecast))

    diffs = [historical[i] - historical[i - 1] for i in range(1, len(historical))]
    try:
        std_dev = statistics.stdev(diffs)
    except statistics.StatisticsError:
        std_dev = 0.0

    lower, upper = [], []
    for h, fv in enumerate(forecast, start=1):
        half_width = z * std_dev * math.sqrt(h)
        lower.append(fv - half_width)
        upper.append(fv + half_width)

    return lower, upper
