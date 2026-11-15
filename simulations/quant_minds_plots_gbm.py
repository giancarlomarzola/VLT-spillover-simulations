"""
GBM single-path price figure for the QuantMinds presentation (Experiment 2), in the same style as the
BTC figures of simulations/quant_minds_plots_btc.py. Saved to data/outputs/figures/quant_minds/:
    gbm_seed<seed>_vol<vol>_price.png: input price and simulated prices, and cumulative spillover.
    summary_table_vol<vol>.tex: the deck's Experiment 2 summary table, laid out like summary_table_btc.tex.

Run from the repo root. Needs the per-seed results in data/outputs/GBM_simulations/<experiment>/iterations/.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.experiment_runner import load_setup, simulate_seed
from utils.paths import FIGURES_DIR, GBM_SIMULATIONS

save_plots = True

output_dir = FIGURES_DIR / "quant_minds"
if save_plots:
    output_dir.mkdir(parents=True, exist_ok=True)

experiment = "1s_2kappas"  # folder in data/outputs/GBM_simulations/ written by run_gbm_experiment.py
single_path_vol = "vol_100"  # resets are too rare at lower volatilities to show one path
single_path_seed = 579  # a path where the inverse product resets (buys) repeatedly

# Same palette, line styles and rc settings as the BTC figures in quant_minds_plots.py
BLACK, BLUE, VERMILLION = "#000000", "#0072B2", "#D55E00"
RED = "#990000"
output_style = {
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
}

experiment_dir = GBM_SIMULATIONS / experiment
paths = sorted((experiment_dir / "iterations").glob("seed_*.parquet"))
if not paths:
    raise FileNotFoundError(f"No saved iterations in {experiment_dir / 'iterations'}; run run_gbm_experiment.py first")
runs = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)

# ---------- summary table at one volatility ----------
# Same statistics and layout as summary_table_btc.tex in quant_minds_plots_btc.py. No GBM run ends
# in a feedback loop or a liquidation, so no run is left out of the averages.
table_vol = "vol_100"
table_runs = runs[runs["price_series"] == table_vol].copy()
table_runs["abs_final"] = table_runs["final_price_deviation"].abs()
# Largest single-trade impact on a path, buy or sell (NaN on paths without resets)
table_runs["max_abs_impact"] = pd.concat(
    [table_runs["max_positive_impact"], -table_runs["max_negative_impact"]], axis=1
).max(axis=1, skipna=True)
active = table_runs[table_runs["orderbook"] == "baseline"]["n_rebalances"] > 0
print(f"Paths with at least one reset at {table_vol} (baseline book): {active.sum()} ({active.mean():.1%})")
# Trade size and impact are NaN on paths without resets, so their means are over the active paths
table = table_runs.groupby("orderbook", sort=False).agg(
    resets=("n_rebalances", "mean"),
    trade_k=("avg_rebalance_size", lambda s: s.mean() / 1e3),
    mean_abs_bps=("avg_abs_impact", lambda s: s.mean() * 1e4),
    max_abs_bps=("max_abs_impact", lambda s: s.mean() * 1e4),
    final_bps=("final_price_deviation", lambda s: s.mean() * 1e4),
    max_abs_final_pct=("abs_final", lambda s: s.max() * 1e2),
)
print(f"\nSummary table ({table_vol}):\n{table.round(3)}")
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
summary_columns = {  # column -> number format
    "resets": "{:.1f}", "trade_k": "{:,.0f}", "mean_abs_bps": "{:.2f}", "max_abs_bps": "{:.2f}",
    "final_bps": "{:.1f}", "max_abs_final_pct": "{:.2f}",
}
summary_lines = [
    # Plain tabular (not tabular* at \textwidth), so the deck's \resizebox can shrink it to fit
    r"\begin{tabular}{@{}lrrrrrr@{}}",
    r"    \toprule",
    r"    & & \textbf{Trade} & \multicolumn{2}{c}{\textbf{Impact (bps)}}"
    r" & \multicolumn{2}{c}{\textbf{Final spillover}}\\",
    r"    \cmidrule(lr){4-5}\cmidrule(lr){6-7}",
    r"    \textbf{Order book} & \textbf{Resets} & (\$k) & Mean $|\gamma_t|$ & Max $|\gamma_t|$"
    r" & Mean (bps) & Max abs.\ (\%)\\",
    r"    \midrule",
]
for book, label in summary_rows.items():
    cells = [label]
    for column, fmt in summary_columns.items():
        value = table.loc[book, column]
        # Without an order book there is no impact or spillover to report
        no_impact = book == "no_orderbook" and column.endswith(("_bps", "_pct"))
        cells.append("--" if no_impact or pd.isna(value) else fmt.format(value))
    summary_lines.append("    " + " & ".join(cells) + r"\\")
summary_lines += [r"    \bottomrule", r"\end{tabular}", ""]
if save_plots:
    table_file = f"summary_table_{table_vol.replace('_', '')}.tex"  # summary_table_vol100.tex
    (output_dir / table_file).write_text("\n".join(summary_lines), encoding="utf-8")

# ---------- one path, re-simulated in full ----------
single_path_books = ["baseline", "depth 0.5x", "spread 2x"]
setup = load_setup(experiment_dir)  # the setup the saved results were run with
setup["orderbooks"] = [ob for ob in setup["orderbooks"] if ob is None or ob.name in single_path_books]
setup["price_generator"].volatilities = [int(single_path_vol.removeprefix("vol_")) / 100]
summary, results = simulate_seed(single_path_seed, **setup, keep_results=True)
# The re-run must reproduce the saved results, or the setup has changed since the experiment ran
saved = runs[(runs["seed"] == single_path_seed) & (runs["price_series"] == single_path_vol)].set_index("orderbook")
for _, row in summary.iterrows():
    expected = saved.loc[row["orderbook"]]
    if row["n_rebalances"] != expected["n_rebalances"] or abs(
        row["final_price_deviation"] - expected["final_price_deviation"]
    ) > 1e-12:
        raise ValueError(f"Re-run of seed {single_path_seed}, {row['orderbook']} does not match the saved results")

res = {book: results[single_path_vol, book] for book in single_path_books}
base = res["baseline"]
hours = base.index.total_seconds() / 3600
step = 20  # plot every 20th second: plenty for a 40-hour axis
file_stem = f"gbm_seed{single_path_seed}_{single_path_vol.replace('_', '')}"
hour_ticks = np.arange(0, 41, 5)
# In legend order: book, label, colour, line width, z-order. The input price is black and drawn on top.
path_styles = [
    ("no_orderbook", "Input price", BLACK, 1.4, 3),
    ("baseline", "Baseline book", BLUE, 1.2, 2),
    ("depth 0.5x", r"Depth 0.5$\times$", RED, 1.2, 2),
    ("spread 2x", r"Spread 2$\times$", VERMILLION, 1.2, 2),
]

with plt.rc_context(output_style):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    ax = axes[0]
    for book, label, color, lw, zorder in path_styles:
        price = base["raw_price"] if book == "no_orderbook" else res[book]["simulated_price"]
        ax.plot(hours[::step], price.iloc[::step], color=color, lw=lw, label=label, zorder=zorder)
    ax.set_title("Input price and simulated price", fontsize=15)
    ax.set_ylabel("Price")
    ax.legend(fontsize=12, loc="best")
    ax = axes[1]
    for book, label, color, lw, zorder in path_styles[1:]:  # spillover is zero for the input price
        ax.plot(hours[::step], (res[book]["price_multiplier"].iloc[::step] - 1) * 100,
                color=color, lw=1.6, label=label, zorder=zorder)
    ax.axhline(0, color=BLACK, lw=1.0, zorder=3)  # the input price, as in the left panel
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
print(f"Seed {single_path_seed}, {single_path_vol} final spillover: "
      + ", ".join(f"{book} {res[book]['price_multiplier'].iloc[-1] - 1:.2%}" for book in single_path_books))
