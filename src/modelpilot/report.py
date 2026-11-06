"""Results table (markdown) and cost-vs-score chart (PNG)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files only; no display needed in WSL
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

# Reference palette (light mode) from the dataviz guidance.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
ROUTER_COLOR = "#2a78d6"
BASELINE_COLOR = "#eb6834"
LLMROUTER_COLOR = "#1baf7a"
FRONTIER_COLOR = "#52514e"


def results_table(baselines: pd.DataFrame, sweep: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy: baselines (strategy, avg_score, total_cost) plus each router lambda."""
    router = pd.DataFrame(
        {
            "strategy": [f"router λ={lam:g}" for lam in sweep["lambda"]],
            "avg_score": sweep["avg_score"],
            "total_cost": sweep["total_cost"],
        }
    )
    table = pd.concat([baselines[["strategy", "avg_score", "total_cost"]], router], ignore_index=True)
    return table.sort_values("total_cost", ignore_index=True)


def to_markdown(table: pd.DataFrame, reference_cost: float) -> str:
    """Markdown table; cost is also shown as a % of `reference_cost` (always-best)."""
    lines = [
        "| Strategy | Avg score | Total cost ($) | Cost vs always-best |",
        "|---|---:|---:|---:|",
    ]
    for row in table.itertuples():
        share = row.total_cost / reference_cost
        lines.append(f"| {row.strategy} | {row.avg_score:.3f} | {row.total_cost:.3f} | {share:.0%} |")
    return "\n".join(lines) + "\n"


def plot_cost_vs_score(baselines: pd.DataFrame, sweep: pd.DataFrame, path: Path) -> None:
    """Router curve over lambda plus baseline points; cost on a log axis."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(
        sweep["total_cost"], sweep["avg_score"], color=ROUTER_COLOR, linewidth=2,
        marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, label="kNN router (λ sweep)",
    )
    # Neighbouring points are close on the log axis, so alternate labels above and below.
    for i, (lam, cost, score) in enumerate(zip(sweep["lambda"], sweep["total_cost"], sweep["avg_score"])):
        dy = -15 if i % 2 == 0 else 9
        ax.annotate(f"λ={lam:g}", (cost, score), xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=8, color=TEXT_SECONDARY)

    ax.scatter(
        baselines["total_cost"], baselines["avg_score"], s=60, color=BASELINE_COLOR,
        edgecolors=SURFACE, linewidths=2, zorder=3, label="Baselines",
    )
    for row in baselines.itertuples():
        # always-best sits at the right edge, so its label goes to the left of the point.
        right_edge = row.strategy.startswith("always-best")
        ax.annotate(row.strategy, (row.total_cost, row.avg_score),
                    xytext=(-8, 6) if right_edge else (8, -14), ha="right" if right_edge else "left",
                    textcoords="offset points", fontsize=9, color=TEXT_PRIMARY)

    ax.set_xscale("log")
    ax.set_xlim(0.04, 3)
    ax.set_ylim(0.68, 0.95)
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Total cost on test split ($, log scale)", color=TEXT_SECONDARY)
    ax.set_ylabel("Avg score", color=TEXT_SECONDARY)
    ax.set_title("ModelPilot: avg score vs cost (test split, 3,729 queries)", color=TEXT_PRIMARY, loc="left")
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.legend(frameon=False, loc="upper right", labelcolor=TEXT_PRIMARY)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def pareto_frontier(table: pd.DataFrame) -> pd.DataFrame:
    """Rows no other row beats on both cost and score: walk by cost, keep each new best score."""
    ordered = table.sort_values(["total_cost", "avg_score"], ascending=[True, False])
    keep = ordered["avg_score"] > ordered["avg_score"].cummax().shift(fill_value=float("-inf"))
    return ordered[keep]


def plot_comparison(table: pd.DataFrame, path: Path) -> None:
    """All rows of results/v4_llmrouter/comparison.csv on one cost-vs-score chart.

    ModelPilot = line over λ; LLMRouter default labels = filled dots; modified labels =
    hollow diamonds; baselines = labelled points. The frontier skips the oracle, which
    sees the answers.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    real = table[table["router"] != "oracle"]
    frontier = pareto_frontier(real)
    ax.plot(frontier["total_cost"], frontier["avg_score"], color=FRONTIER_COLOR, linewidth=1,
            drawstyle="steps-post", zorder=1, label="Pareto frontier (excl. oracle)")

    ours = table[table["source"] == "ModelPilot"]
    ax.plot(ours["total_cost"], ours["avg_score"], color=ROUTER_COLOR, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, label="ModelPilot kNN (λ sweep)")

    llm = table[table["source"] == "LLMRouter"]
    default, modified = llm[llm["labels"] == "default"], llm[llm["labels"] == "modified labels"]
    ax.scatter(default["total_cost"], default["avg_score"], s=36, color=LLMROUTER_COLOR, edgecolors=SURFACE,
               linewidths=1.5, zorder=2, label="LLMRouter, default labels (α sweep)")
    ax.scatter(modified["total_cost"], modified["avg_score"], s=70, marker="D", facecolors="none",
               edgecolors=LLMROUTER_COLOR, linewidths=2, zorder=4, label="LLMRouter, modified labels (α=1)")

    base = table[table["source"] == "baseline"]
    ax.scatter(base["total_cost"], base["avg_score"], s=60, color=BASELINE_COLOR, edgecolors=SURFACE,
               linewidths=2, zorder=4, label="Baselines")
    for row in base.itertuples():
        right_edge = row.router.startswith("always-best")
        ax.annotate(row.router, (row.total_cost, row.avg_score), xytext=(-8, -22) if right_edge else (8, -14),
                    ha="right" if right_edge else "left", textcoords="offset points", fontsize=8, color=TEXT_PRIMARY)

    # Name the default-label LLMRouter points on the frontier; the rest stay unlabelled context.
    on_frontier = frontier[(frontier["source"] == "LLMRouter") & (frontier["labels"] == "default")]
    for row in on_frontier.itertuples():
        ax.annotate(f"{row.router} {row.setting}", (row.total_cost, row.avg_score), xytext=(6, 6),
                    textcoords="offset points", fontsize=7, color=TEXT_SECONDARY)
    # The modified-label rows sit on top of each other (and of always-best), so they share one label.
    if not modified.empty:
        names = " & ".join(sorted(modified["router"]))
        ax.annotate(f"{names}, modified labels", (modified["total_cost"].min(), modified["avg_score"].max()),
                    xytext=(-8, 10), ha="right", textcoords="offset points", fontsize=7, color=TEXT_SECONDARY)

    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.03, 3)
    ax.set_xlabel("Total cost on test split ($, log scale)", color=TEXT_SECONDARY)
    ax.set_ylabel("Avg score", color=TEXT_SECONDARY)
    ax.set_title("ModelPilot vs LLMRouter: avg score vs cost (test split, MiniLM embeddings)",
                 color=TEXT_PRIMARY, loc="left", fontsize=11)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.legend(frameon=False, loc="upper left", fontsize=8, labelcolor=TEXT_PRIMARY)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
