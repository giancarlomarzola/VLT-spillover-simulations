"""
GBM figures for the QuantMinds presentation (Experiment 2), in the same style as the BTC figures of
simulations/quant_minds_plots_btc.py. The deck shows two books only, the baseline and depth 0.5x, against
the no-impact reference, and compares them across volatilities. Saved to data/outputs/figures/quant_minds/:
    gbm_resets_by_vol.png: share of paths with a reset, and resets per path with a reset, by volatility.
    summary_table_gbm_by_vol.tex: the deck's Experiment 2 summary table, one block of rows per volatility.
    gbm_seed<seed>_vol<vol>_price.png: input price and simulated prices, and cumulative spillover.

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

# Same palette, line styles and rc settings as the BTC figures in quant_minds_plots_btc.py
BLACK, GREY, BLUE = "#000000", "#999999", "#0072B2"
RED = "#990000"
NAVY = "#142D50"  # presentation colour, for single-series panels where colour only decorates
output_style = {
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
}
# The two books in the deck, with the no-impact reference first: label, colour, marker
books = {
    "no_orderbook": ("No order book", GREY, "s"),
    "baseline": ("Baseline", BLUE, "o"),
    "depth 0.5x": (r"Depth 0.5$\times$", RED, "D"),
}
impact_books = ["baseline", "depth 0.5x"]

experiment_dir = GBM_SIMULATIONS / experiment
paths = sorted((experiment_dir / "iterations").glob("seed_*.parquet"))
if not paths:
    raise FileNotFoundError(f"No saved iterations in {experiment_dir / 'iterations'}; run run_gbm_experiment.py first")
runs = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
runs = runs[runs["orderbook"].isin(books)].copy()
runs["vol"] = runs["price_series"].str.removeprefix("vol_").astype(int)
runs["active"] = runs["n_rebalances"] > 0  # paths with at least one reset
# Largest spillover |m_t - 1| anywhere on a path, up or down
runs["max_abs_spillover"] = pd.concat([runs["max_price_deviation"], -runs["min_price_deviation"]], axis=1).max(axis=1)
vols = sorted(runs["vol"].unique())
n_paths = runs["seed"].nunique()
print(f"{n_paths} iterations from {experiment_dir / 'iterations'}")

# Whether a path resets at all is decided by the input price before any impact, so the same paths are
# active on every book. Check it, so "paths with a reset" means one set of paths in every panel.
active_sets = runs[runs["active"]].groupby(["vol", "orderbook"])["seed"].apply(frozenset).unstack()
if not active_sets.apply(lambda row: row.nunique() == 1, axis=1).all():
    raise ValueError("Paths with a reset differ between books; the conditional statistics need a per-book count")
share_active = runs[runs["orderbook"] == "baseline"].groupby("vol")["active"].agg(["sum", "mean"])
print(f"\nPaths with at least one reset:\n{share_active}")
reset_vols = [vol for vol in vols if share_active.loc[vol, "sum"] > 0]  # volatilities with any reset
active = runs[runs["active"]]

# ---------- summary table, by volatility ----------
# Statistics on the paths with a reset: trade size and impact are NaN on the others, and averaging
# resets or spillover over paths that never trade only measures how rare resets are (left panel).
table = active.groupby(["vol", "orderbook"], sort=False).agg(
    resets=("n_rebalances", "mean"),
    trade_k=("avg_rebalance_size", lambda s: s.mean() / 1e3),
    mean_abs_bps=("avg_abs_impact", lambda s: s.mean() * 1e4),
    max_spillover_pct=("max_abs_spillover", lambda s: s.max() * 1e2),
)
print(f"\nSummary table (paths with a reset):\n{table.round(3)}")
summary_columns = {  # column -> number format
    "resets": "{:.1f}", "trade_k": "{:,.0f}", "mean_abs_bps": "{:.2f}", "max_spillover_pct": "{:.2f}",
}
summary_lines = [
    # Plain tabular at its natural width (not tabular* at \textwidth); the deck sets the font size
    r"\begin{tabular}{@{}llcccc@{}}",
    r"    \toprule",
    r"    $\sigma$ & \textbf{Order book} & \textbf{Resets} & Mean $|q_t|$ & Mean $|\gamma_t|$"
    r" & Max $|m_t-1|$\\",
    r"    & & & (\$k) & (bps) & (\%)\\",  # units, in their own row
]
for vol in reset_vols:
    summary_lines.append(r"    \midrule")
    n_active = share_active.loc[vol, "sum"]
    # Volatility on the first row of its block, the number of paths with a reset on the second
    first_cells = [rf"{vol}\%", rf"\scriptsize({n_active:,} path{'' if n_active == 1 else 's'})", ""]
    for first, (book, (label, _, _)) in zip(first_cells, books.items()):
        cells = [first, label.replace("0.5$\\times$", "$0.5\\times$")]  # same label as summary_table_btc.tex
        for column, fmt in summary_columns.items():
            value = table.loc[(vol, book), column]
            # Without an order book there is no impact or spillover to report
            no_impact = book == "no_orderbook" and column != "resets" and column != "trade_k"
            cells.append("--" if no_impact or pd.isna(value) else fmt.format(value))
        summary_lines.append("    " + " & ".join(cells) + r"\\")
summary_lines += [r"    \bottomrule", r"\end{tabular}", ""]
if save_plots:
    (output_dir / "summary_table_gbm_by_vol.tex").write_text("\n".join(summary_lines), encoding="utf-8")


def vol_ticklabels(vol_list):
    """Volatility tick labels, with the number of paths with a reset beneath."""
    return [f"{vol}%\n({share_active.loc[vol, 'sum']:,} paths)" for vol in vol_list]


# ---------- resets by volatility ----------
with plt.rc_context(output_style):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    x = np.arange(len(vols))
    shares = share_active["mean"].reindex(vols) * 100
    ax.bar(x, shares, width=0.6, color=NAVY, zorder=2)
    for xi, share in zip(x, shares):  # few bars, and the small values are the point: label them all
        ax.text(xi, share + 0.15, f"{share:.1f}%", ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x, [f"{vol}%" for vol in vols])
    ax.set_ylim(0, shares.max() * 1.18)
    ax.set_title("Paths with at least one reset", fontsize=15)
    ax.set_ylabel(r"Share of $1{,}000$ paths (%)")
    ax.set_xlabel("Annualised GBM volatility")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    x = np.arange(len(reset_vols))
    width = 0.26
    for i, (book, (label, color, _)) in enumerate(books.items()):
        resets = [table.loc[(vol, book), "resets"] for vol in reset_vols]
        ax.bar(x + (i - 1) * width, resets, width=width * 0.92, color=color, label=label, zorder=2)
    ax.set_xticks(x, vol_ticklabels(reset_vols))
    ax.set_title("Resets per path with a reset", fontsize=15)
    ax.set_ylabel("Mean resets per path")
    ax.set_xlabel("Annualised GBM volatility")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=12, loc="upper left")
    fig.tight_layout()
    if save_plots:
        fig.savefig(output_dir / "gbm_resets_by_vol.png", dpi=300, bbox_inches="tight")
    else:
        fig.show()

# ---------- one path, re-simulated in full ----------
single_path_books = impact_books
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
]

with plt.rc_context(output_style):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    ax = axes[0]
    for book, label, color, lw, zorder in path_styles:
        price = base["raw_price"] if book == "no_orderbook" else res[book]["simulated_price"]
        ax.plot(hours[::step], price.iloc[::step], color=color, lw=lw, label=label, zorder=zorder)
    ax.set_title("Input price and simulated price", fontsize=15)
    ax.set_ylabel(r"Price (\$)")
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
