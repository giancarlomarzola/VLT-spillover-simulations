"""
Figures for the QuantMinds presentation, saved to data/outputs/figures/quant_minds/.

1. Order book scenarios: one plot per order book, drawn from its parameters.
2. Simulation outputs, from the BTC bootstrap experiment saved by run_btc_bootstrap_experiment.py:
     btc_summary.png: mean absolute final spillover and mean resets by order book.
     btc_seed<seed>_price.png / _leverage.png: one path, re-simulated in full here.
     summary_table_btc.tex: the deck's summary table.

Run from the repo root. Step 2 needs the per-seed results in
data/outputs/BTC_bootstrap_simulations/<experiment>/iterations/.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.experiment_runner import load_setup, simulate_seed
from core.simulation import flag_feedback_loops
from simulations.products_and_orderbooks import btc_orderbooks
from utils.paths import BOOTSTRAP_SIMULATIONS, FIGURES_DIR
from utils.plotting import plot_orderbooks

save_plots = True

output_dir = FIGURES_DIR / "quant_minds"
if save_plots:
    output_dir.mkdir(parents=True, exist_ok=True)


# ========================================
# 1. Order book scenarios
# ========================================

# The BTC books of the bootstrap experiment (simulations.products_and_orderbooks): the baseline and
# the baseline with one parameter changed at a time. Half-life does not change a static book, so it
# has no plot of its own, and the run without an order book (None) has nothing to draw.
orderbooks_list = [
    ob for ob in btc_orderbooks() if ob is not None and ob.name != "resilience 30min"
]
baseline_orderbook = orderbooks_list[0]

for orderbook in orderbooks_list:
    fig = plot_orderbooks(
        orderbook,
        show_parameters=False, show_legend=False,
        x_limit=2.2, y_limit=60_000_000,
        figsize=(5.5, 3.1),  # small canvas = large text relative to the plot (saved at 300 dpi)
        baseline=baseline_orderbook,  # dotted outline; skipped on the baseline's own plot
        title="",
        )
    if save_plots:
        file_name = orderbook.name.replace(".", "_")  # graphicx misreads extra dots in a file name
        fig.savefig(output_dir / f"orderbook_{file_name}.png", dpi=300, bbox_inches="tight")
    else:
        fig.show()


# ========================================
# 2. Simulation outputs
# ========================================

experiment = "100ms_4h_block1s"  # folder in data/outputs/BTC_bootstrap_simulations/ written by run_btc_bootstrap_experiment.py
# Cherry-picked for illustrative purposes. 841: price rises, only BTCDOWN resets (buys).
# 619: price falls, both products reset (both sell); the depth 0.5x book ends in a feedback loop.
# Override from the command line: python simulations/quant_minds_plots.py 619
single_path_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 619

# Okabe-Ito palette: distinguishable under the common forms of colour blindness
BLACK, GREY = "#000000", "#999999"
ORANGE, SKY_BLUE, GREEN, BLUE, VERMILLION, PINK = "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"
# Presentation colours, for plots where colour only decorates (one series per panel)
NAVY, RED = "#142D50", "#990000"
output_style = {
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
}
# label, colour, line style, marker, line width
book_styles = {
    "baseline": ("Baseline", BLACK, "-", "o", 2.6),
    "resilience 30min": ("Half-life 30 min", SKY_BLUE, "--", "s", 1.8),
    "depth 0.5x": (r"Depth 0.5$\times$", PINK, "--", "v", 1.8),
    "width 2x": (r"Width 2$\times$", ORANGE, "-", "D", 1.8),
    "shape k1": (r"Shape $\kappa=1$", BLUE, "-", "^", 1.8),
    "shape k1_5": (r"Shape $\kappa=1.5$", GREEN, ":", "P", 1.8),
    "spread 2x": (r"Spread 2$\times$", VERMILLION, "-", "o", 2.6),
}

experiment_dir = BOOTSTRAP_SIMULATIONS / experiment
results_dir = experiment_dir / "iterations"
paths = sorted(results_dir.glob("seed_*.parquet"))
if not paths:
    raise FileNotFoundError(f"No saved iterations in {results_dir}; run run_btc_bootstrap_experiment.py first")
runs = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
runs["active"] = runs["n_rebalances"] > 0  # paths with at least one reset
runs["abs_final"] = runs["final_price_deviation"].abs()
# Largest single-trade impact on a path, buy or sell (NaN on paths without resets)
runs["max_abs_impact"] = pd.concat(
    [runs["max_positive_impact"], -runs["max_negative_impact"]], axis=1
).max(axis=1, skipna=True)
# Feedback loops are counted, but left out of every average (as in core.summary_tables): their
# price runs off to the floor and would swamp the other runs
runs["loop"] = flag_feedback_loops(runs)["feedback_loop"]
print(f"{runs['seed'].nunique()} iterations from {results_dir}")
print(f"Feedback loops:\n{runs[runs['loop']][['seed', 'orderbook', 'final_price_deviation']]}")
clean = runs[~runs["loop"]]

pd.set_option("display.width", 250, "display.max_columns", 50)
active = runs[runs["orderbook"] == "baseline"]["active"]
print(f"\nPaths with at least one reset (baseline book): {active.sum()} ({active.mean():.1%})")

# ---------- numbers for the slides ----------
# Trade size and impact are NaN on paths without resets, so their means are over the active paths
table = clean.groupby("orderbook", sort=False).agg(
    resets=("n_rebalances", "mean"),
    trade_k=("avg_rebalance_size", lambda s: s.mean() / 1e3),
    mean_abs_bps=("avg_abs_impact", lambda s: s.mean() * 1e4),
    max_abs_bps=("max_abs_impact", lambda s: s.mean() * 1e4),
    final_bps=("final_price_deviation", lambda s: s.mean() * 1e4),
    max_abs_final_pct=("abs_final", lambda s: s.max() * 1e2),
)
table["loops"] = runs.groupby("orderbook", sort=False)["loop"].sum()
print(f"\nSummary table:\n{table.round(3)}")
# Slide version of the table: a tabular to \input inside the frame, which sets the font size and
# \tabcolsep. Needs booktabs in the deck, and a \red{} macro if any row is highlighted.
summary_rows = {  # row label in the deck, for the books in the order shown
    "no_orderbook": "No order book",
    "baseline": "Baseline",
    "depth 0.5x": r"Depth $0.5\times$",
    "width 2x": r"Width $2\times$",
    "spread 2x": r"Spread $2\times$",
    "shape k1": r"Shape $\kappa=1$",
    "shape k1_5": r"Shape $\kappa=1.5$",
    "resilience 30min": "Half-life 30 min",
}
highlighted_rows = set()  # books whose row is wrapped in \red{}, e.g. {"spread 2x"}
summary_columns = {  # column -> number format
    "resets": "{:.1f}", "trade_k": "{:,.0f}", "mean_abs_bps": "{:.2f}", "max_abs_bps": "{:.2f}",
    "final_bps": "{:.1f}", "max_abs_final_pct": "{:.2f}", "loops": "{:.0f}",
}
summary_lines = [
    # Plain tabular (not tabular* at \textwidth), so the deck's \resizebox can shrink it to fit
    r"\begin{tabular}{@{}lrrrrrrr@{}}",
    r"    \toprule",
    r"    & & \textbf{Trade} & \multicolumn{2}{c}{\textbf{Impact (bps)}}"
    r" & \multicolumn{2}{c}{\textbf{Final spillover}} & \textbf{Feedback}\\",
    r"    \cmidrule(lr){4-5}\cmidrule(lr){6-7}",
    r"    \textbf{Order book} & \textbf{Resets} & (\$k) & Mean $|\gamma_t|$ & Max $|\gamma_t|$"
    r" & Mean (bps) & Max abs.\ (\%) & \textbf{loops}\\",
    r"    \midrule",
]
for book, label in summary_rows.items():
    cells = [label]
    for column, fmt in summary_columns.items():
        value = table.loc[book, column]
        # Without an order book there is no impact or spillover to report
        no_impact = book == "no_orderbook" and column.endswith(("_bps", "_pct"))
        cells.append("--" if no_impact or pd.isna(value) else fmt.format(value))
    if book in highlighted_rows:
        cells = [rf"\red{{{cell}}}" for cell in cells]
    summary_lines.append("    " + " & ".join(cells) + r"\\")
summary_lines += [r"    \bottomrule", r"\end{tabular}", ""]
if save_plots:
    (output_dir / "summary_table_btc.tex").write_text("\n".join(summary_lines), encoding="utf-8")

# ---------- spillover and resets by order book ----------
# Means with 95% confidence intervals, normal approximation: half-width 1.96 * std / sqrt(n), over
# the runs without a feedback loop. The final spillover is taken in absolute value: the bootstrapped
# paths go up as often as down, so signed spillovers largely cancel out in the mean.
stats = clean.groupby("orderbook").agg(
    final_mean=("abs_final", "mean"), final_std=("abs_final", "std"),
    resets_mean=("n_rebalances", "mean"), resets_std=("n_rebalances", "std"), n=("seed", "size"),
)
half_width = 1.96 / np.sqrt(stats["n"])
summary_books = {"no_orderbook": ("No order book", GREY, "-", "o", 1.8), **book_styles}
order = list(summary_books)[::-1]  # top to bottom: no order book, baseline, then the scenarios
with plt.rc_context(output_style):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), sharey=True)
    for y, book in enumerate(order):
        label, color, ls, marker, lw = summary_books[book]
        s = stats.loc[book]
        style = dict(color=color, marker=marker, ms=8, elinewidth=1.6, capsize=4, ls="none")
        if book != "no_orderbook":  # no spillover without price impact
            axes[0].errorbar(s["final_mean"] * 1e4, y, xerr=s["final_std"] * half_width[book] * 1e4, **style)
        axes[1].errorbar(s["resets_mean"], y, xerr=s["resets_std"] * half_width[book], **style)
    axes[0].set_yticks(range(len(order)), [summary_books[book][0] for book in order])
    axes[0].set_title("Absolute spillover after 4 hours", fontsize=15)
    axes[0].set_xlabel("Mean absolute final spillover (bps)")
    axes[1].set_title("Number of resets", fontsize=15)
    axes[1].set_xlabel("Mean resets per path")
    for ax in axes:
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    if save_plots:
        fig.savefig(output_dir / "btc_summary.png", dpi=300, bbox_inches="tight")
    else:
        fig.show()

# ---------- one path, re-simulated in full ----------
single_path_books = ["baseline", "depth 0.5x", "spread 2x"]
setup = load_setup(experiment_dir)  # the setup the saved results were run with
btc_up, btc_down = setup["vletps"]
setup["orderbooks"] = [ob for ob in setup["orderbooks"] if ob is not None and ob.name in single_path_books]
summary, results = simulate_seed(single_path_seed, **setup, keep_results=True)
# The re-run must reproduce the saved results, or the setup has changed since the experiment ran
saved = runs[runs["seed"] == single_path_seed].set_index("orderbook")
for _, row in summary.iterrows():
    expected = saved.loc[row["orderbook"]]
    if row["n_rebalances"] != expected["n_rebalances"] or abs(
        row["final_price_deviation"] - expected["final_price_deviation"]
    ) > 1e-12:
        raise ValueError(f"Re-run of seed {single_path_seed}, {row['orderbook']} does not match the saved results")

price_label = summary["price_series"].iloc[0]
res = {book: results[price_label, book] for book in single_path_books}
base = res["baseline"]
hours = base.index.total_seconds() / 3600
step = 10  # plot every 10th step (once a second): plenty for a 4-hour axis
file_stem = f"btc_seed{single_path_seed}"
hour_ticks = np.arange(0, 4.01, 0.5)

with plt.rc_context(output_style):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    ax = axes[0]
    ax.plot(hours[::step], base["raw_price"].iloc[::step], color=GREY, lw=1.2, label="Bootstrap input\n(no impact)")
    ax.plot(hours[::step], base["simulated_price"].iloc[::step], color=BLACK, lw=1.2, label="Baseline book")
    ax.plot(hours[::step], res["spread 2x"]["simulated_price"].iloc[::step], color=VERMILLION, lw=1.2,
            label=r"Spread 2$\times$")
    ax.plot(hours[::step], res["depth 0.5x"]["simulated_price"].iloc[::step], color=PINK, lw=1.6, ls="--",
            label=r"Depth 0.5$\times$", zorder=1.5)  # drawn under the other paths, listed last in the legend
    ax.set_title("Input price and simulated price", fontsize=15)
    ax.set_ylabel("BTC price (US\\$)")
    # Lower left, just above the zero line: clear of the paths on seed 619 (adjust for other seeds)
    ax.legend(fontsize=12, loc="lower left", bbox_to_anchor=(0.0, 0.07))
    ax = axes[1]
    for book, label, color, ls, zorder in [
        ("baseline", "Baseline book", BLACK, "-", 2),
        ("spread 2x", r"Spread 2$\times$", VERMILLION, "-", 2),
        ("depth 0.5x", r"Depth 0.5$\times$", PINK, "--", 1.5),  # under the others, as in the price panel
    ]:
        ax.plot(hours[::step], (res[book]["price_multiplier"].iloc[::step] - 1) * 100,
                color=color, lw=1.6, ls=ls, label=label, zorder=zorder)
    ax.axhline(0, color=GREY, lw=0.8)  # grey, so it is not mistaken for the black baseline
    ax.set_title("Cumulative spillover from rebalancing", fontsize=15)
    ax.set_ylabel("Deviation from input price (%)")
    ax.legend(fontsize=12, loc="best")
    for ax in axes:
        ax.set_xlabel("Time elapsed (hours)")
        ax.set_xticks(hour_ticks)
    fig.tight_layout()
    if save_plots:
        fig.savefig(output_dir / f"{file_stem}_price.png", dpi=300, bbox_inches="tight")
    else:
        fig.show()

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.0), sharey=True)
    for ax, product, name, color in [
        (axes[0], btc_up, "Direct (BTCUP)", NAVY),
        (axes[1], btc_down, "Inverse (BTCDOWN)", RED),
    ]:
        delta = base[f"actual_delta_{product.name}"]
        leverage = base[f"leverage_after_{product.name}"].abs()
        n_resets = int((delta != 0).sum())
        ax.plot(hours[::step], leverage.iloc[::step], color=color, lw=1.2)
        for bound in (abs(product.leverage_lower_bound), abs(product.leverage_upper_bound)):
            ax.axhline(bound, color="black", lw=1.0, ls="--")
        ax.set_title(f"{name}: {n_resets:,} reset{'' if n_resets == 1 else 's'}", fontsize=15)
        ax.set_xlabel("Time elapsed (hours)")
        ax.set_xticks(hour_ticks)
        ax.set_ylim(1.0, 4.3)
        at_upper = int(((delta != 0) & (leverage > abs(product.leverage_upper_bound) - 1e-6)).sum())
        print(f"{name}: {n_resets} resets ({at_upper} at the upper bound), "
              f"{int((delta > 0).sum())} purchases, {int((delta < 0).sum())} sales")
    axes[0].set_ylabel(r"Leverage after any reset, $|\lambda_t^*|$")
    fig.tight_layout()
    if save_plots:
        fig.savefig(output_dir / f"{file_stem}_leverage.png", dpi=300, bbox_inches="tight")
    else:
        fig.show()
print(f"Seed {single_path_seed} final spillover: "
      + ", ".join(f"{book} {res[book]['price_multiplier'].iloc[-1] - 1:.2%}" for book in single_path_books))
