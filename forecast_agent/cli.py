"""
Command-line interface for the Forecast Agent.

Usage example:

    forecast data.csv \
        --date-col date \
        --value-col revenue \
        --periods 12 \
        --method sma \
        --output forecast.md
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@click.command()
@click.argument("data_csv", metavar="<data.csv>")
@click.option(
    "--date-col",
    default="date",
    show_default=True,
    help="Name of the date/time column in the CSV.",
)
@click.option(
    "--value-col",
    default="value",
    show_default=True,
    help="Name of the numeric column to forecast.",
)
@click.option(
    "--periods",
    default=12,
    show_default=True,
    type=int,
    help="Number of future periods to forecast.",
)
@click.option(
    "--method",
    default="sma",
    show_default=True,
    type=click.Choice(["sma", "linear_trend", "exponential_smoothing"]),
    help="Forecasting method.",
)
@click.option(
    "--output",
    "output_path",
    default="forecast.md",
    show_default=True,
    help="Path to write the Markdown report.",
)
@click.option(
    "--api-key",
    envvar="ANTHROPIC_API_KEY",
    default=None,
    help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var).",
)
def forecast(
    data_csv: str,
    date_col: str,
    value_col: str,
    periods: int,
    method: str,
    output_path: str,
    api_key: str | None,
) -> None:
    """Run the Forecast Agent on <data.csv> and write a Markdown report."""
    # Lazy import so the CLI starts fast even if anthropic isn't installed.
    try:
        from forecast_agent.agent import ForecastAgent
    except ImportError as exc:
        console.print(f"[red]Import error:[/red] {exc}")
        sys.exit(1)

    console.print(
        Panel(
            f"[bold]Forecast Agent[/bold]\n"
            f"  Input : [cyan]{data_csv}[/cyan]\n"
            f"  Column: [cyan]{value_col}[/cyan] (date: [cyan]{date_col}[/cyan])\n"
            f"  Method: [cyan]{method}[/cyan] | Periods: [cyan]{periods}[/cyan]\n"
            f"  Output: [cyan]{output_path}[/cyan]",
            title="Starting",
            expand=False,
        )
    )

    agent = ForecastAgent(api_key=api_key)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Running forecast agent...", total=None)
        try:
            reply = agent.run(
                csv_path=data_csv,
                date_col=date_col,
                value_col=value_col,
                periods=periods,
                output_path=output_path,
                method=method,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Agent error:[/red] {exc}")
            sys.exit(1)

    console.print("\n[bold green]Agent reply:[/bold green]")
    console.print(reply)
    console.print(f"\n[bold]Report saved to:[/bold] [cyan]{output_path}[/cyan]")


def main() -> None:  # entry-point wrapper
    forecast()


if __name__ == "__main__":
    main()
