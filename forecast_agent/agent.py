"""
ForecastAgent: Anthropic-powered time series forecasting agent.

Uses claude-sonnet-4-6 with a manual tool-use loop.
Tools: read_csv, compute_forecast, write_file.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd

from .models import (
    compute_confidence_interval,
    exponential_smoothing,
    linear_trend,
    simple_moving_average,
)

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a time-series forecasting analyst.
    Your workflow:
    1. Use `read_csv` to load and inspect the data.
    2. Use `compute_forecast` to generate point estimates, confidence intervals,
       and scenario narratives (optimistic / base / pessimistic).
    3. Use `write_file` to persist the Markdown report at the requested output path.
    4. Reply with a brief summary of what you found and where the report was saved.

    Always interpret results for a non-technical audience.
    Cite the forecasting method you used and explain its assumptions.
""")

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_csv",
        "description": (
            "Read a CSV file and return a JSON preview of the data "
            "including column names, row count, and a sample of rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the CSV file.",
                },
                "date_col": {
                    "type": "string",
                    "description": "Name of the date/time column.",
                },
                "value_col": {
                    "type": "string",
                    "description": "Name of the numeric value column to forecast.",
                },
            },
            "required": ["path", "date_col", "value_col"],
        },
    },
    {
        "name": "compute_forecast",
        "description": (
            "Compute a time-series forecast using the loaded data. "
            "Returns point estimates, 80 %% confidence intervals, and "
            "optimistic / base / pessimistic scenario narratives."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the CSV file (same as used in read_csv).",
                },
                "date_col": {
                    "type": "string",
                    "description": "Date column name.",
                },
                "value_col": {
                    "type": "string",
                    "description": "Numeric value column name.",
                },
                "periods": {
                    "type": "integer",
                    "description": "Number of future periods to forecast.",
                },
                "method": {
                    "type": "string",
                    "enum": ["sma", "linear_trend", "exponential_smoothing"],
                    "description": (
                        "Forecasting method: "
                        "'sma' = simple moving average, "
                        "'linear_trend' = OLS linear trend extrapolation, "
                        "'exponential_smoothing' = Holt single exponential smoothing."
                    ),
                },
            },
            "required": ["path", "date_col", "value_col", "periods", "method"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text content to a file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to write to.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write.",
                },
            },
            "required": ["path", "content"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _tool_read_csv(path: str, date_col: str, value_col: str) -> dict[str, Any]:
    """Load the CSV and return a JSON-serialisable summary."""
    df = pd.read_csv(path)
    missing_cols = [c for c in [date_col, value_col] if c not in df.columns]
    if missing_cols:
        return {
            "error": f"Columns not found: {missing_cols}. Available: {list(df.columns)}"
        }
    series = df[value_col].dropna()
    return {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "date_col": date_col,
        "value_col": value_col,
        "value_stats": {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
        },
        "sample_rows": df[[date_col, value_col]].head(5).to_dict(orient="records"),
    }


def _tool_compute_forecast(
    path: str,
    date_col: str,
    value_col: str,
    periods: int,
    method: str,
) -> dict[str, Any]:
    """Run the forecast and return structured results."""
    df = pd.read_csv(path)
    missing_cols = [c for c in [date_col, value_col] if c not in df.columns]
    if missing_cols:
        return {
            "error": f"Columns not found: {missing_cols}. Available: {list(df.columns)}"
        }

    values = df[value_col].dropna().tolist()
    if len(values) < 3:
        return {"error": "Need at least 3 data points to compute a forecast."}

    # --- point forecast ---
    method = method.lower()
    if method == "sma":
        forecast_values = simple_moving_average(values, periods=periods)
        method_name = "Simple Moving Average"
        method_notes = (
            "Uses the mean of recent observations as the forecast. "
            "Assumes no trend or seasonality."
        )
    elif method == "linear_trend":
        forecast_values = linear_trend(values, periods=periods)
        method_name = "Linear Trend (OLS)"
        method_notes = (
            "Fits a straight-line trend to historical data and extrapolates forward. "
            "Assumes a constant rate of change."
        )
    elif method == "exponential_smoothing":
        forecast_values = exponential_smoothing(values, periods=periods)
        method_name = "Single Exponential Smoothing"
        method_notes = (
            "Weights recent observations more heavily than older ones. "
            "Adapts to level changes but does not capture trend."
        )
    else:
        return {"error": f"Unknown method '{method}'. Choose sma, linear_trend, or exponential_smoothing."}

    # --- confidence intervals ---
    ci_lower, ci_upper = compute_confidence_interval(values, forecast_values)

    # --- scenario multipliers ---
    opt_factor = 1.15   # +15 %
    pes_factor = 0.85   # -15 %
    optimistic = [round(v * opt_factor, 4) for v in forecast_values]
    pessimistic = [round(v * pes_factor, 4) for v in forecast_values]
    base = [round(v, 4) for v in forecast_values]

    # --- infer future period labels ---
    try:
        dates = pd.to_datetime(df[date_col])
        freq = pd.infer_freq(dates)
        if freq is None:
            freq = "ME"  # fall back to month-end
        last_date = dates.iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
        period_labels = [d.strftime("%Y-%m-%d") for d in future_dates]
    except Exception:  # noqa: BLE001
        period_labels = [f"t+{i}" for i in range(1, periods + 1)]

    forecast_table = [
        {
            "period": period_labels[i],
            "pessimistic": pessimistic[i],
            "base": base[i],
            "optimistic": optimistic[i],
            "ci_lower_80": round(ci_lower[i], 4),
            "ci_upper_80": round(ci_upper[i], 4),
        }
        for i in range(periods)
    ]

    last_value = values[-1]

    return {
        "method": method_name,
        "method_notes": method_notes,
        "historical_periods": len(values),
        "forecast_periods": periods,
        "last_historical_value": round(last_value, 4),
        "scenarios": {
            "optimistic_assumption": f"+15% above base forecast",
            "base_assumption": f"Continues {method_name.lower()} trajectory",
            "pessimistic_assumption": "-15% below base forecast",
        },
        "forecast_table": forecast_table,
    }


def _tool_write_file(path: str, content: str) -> dict[str, Any]:
    """Write content to disk, creating parent dirs as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return {"written": str(out.resolve()), "bytes": len(content.encode())}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _dispatch(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Call the right tool function and return a JSON string result."""
    try:
        if tool_name == "read_csv":
            result = _tool_read_csv(**tool_input)
        elif tool_name == "compute_forecast":
            result = _tool_compute_forecast(**tool_input)
        elif tool_name == "write_file":
            result = _tool_write_file(**tool_input)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
    except Exception as exc:  # noqa: BLE001
        result = {"error": str(exc)}
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ForecastAgent:
    """
    Anthropic-powered forecasting agent.

    Parameters
    ----------
    api_key:
        Anthropic API key. Defaults to the ``ANTHROPIC_API_KEY`` environment
        variable.
    max_iterations:
        Safety cap on the tool-use loop (default 20).
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_iterations: int = 20,
    ) -> None:
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self._max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        csv_path: str,
        date_col: str,
        value_col: str,
        periods: int,
        output_path: str,
        method: str = "sma",
    ) -> str:
        """
        Run the forecasting agent end-to-end.

        Parameters
        ----------
        csv_path:
            Path to the CSV file containing the time series.
        date_col:
            Name of the date column.
        value_col:
            Name of the numeric column to forecast.
        periods:
            Number of future periods to forecast.
        output_path:
            Where to write the Markdown report.
        method:
            One of ``sma``, ``linear_trend``, ``exponential_smoothing``.

        Returns
        -------
        str
            The final text reply from the model.
        """
        user_message = (
            f"Please forecast the `{value_col}` column in `{csv_path}` "
            f"for the next {periods} periods using the '{method}' method. "
            f"The date column is `{date_col}`. "
            f"Save the full Markdown report to `{output_path}`."
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]

        for _ in range(self._max_iterations):
            response = self._client.messages.create(
                model=MODEL,
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
            )

            # Append assistant turn (content blocks) verbatim.
            messages.append({"role": "assistant", "content": response.content})  # type: ignore[arg-type]

            if response.stop_reason == "end_turn":
                # Extract final text reply.
                final_text = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                return final_text

            if response.stop_reason != "tool_use":
                # Unexpected stop — return whatever text we have.
                return next(
                    (b.text for b in response.content if b.type == "text"),
                    f"[Agent stopped with reason: {response.stop_reason}]",
                )

            # Execute all tool calls and collect results.
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result_str = _dispatch(block.name, block.input)  # type: ignore[arg-type]
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    }
                )

            messages.append({"role": "user", "content": tool_results})  # type: ignore[arg-type]

        return "[Agent reached the maximum iteration limit without completing.]"
