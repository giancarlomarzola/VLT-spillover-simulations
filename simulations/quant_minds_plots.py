"""
Figures for the QuantMinds presentation, saved to data/outputs/figures/quant_minds/.

1. Order book scenarios: one plot per order book, drawn from its parameters.
2. Simulation outputs, from the experiment saved by simulation_from_gbm.py:
     gbm_updown_summary.png: mean final spillover and resets by volatility and order book.
     gbm_seed<seed>_vol<vol>_price.png / _leverage.png: one path, re-simulated in full here.
   It also prints the numbers in the deck's summary table.

Run from the repo root. Step 2 needs the per-seed results in
data/outputs/GBM_simulations/<experiment>/iterations/.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.gbm_iterations import simulate_seed
from core.orderbook import Orderbook
from simulations.simulation_from_gbm import gbm_down, gbm_up, orderbooks, setup
from utils.paths import FIGURES_DIR, GBM_SIMULATIONS
from utils.plotting import plot_orderbooks

save_plots = True

output_dir = FIGURES_DIR / "quant_minds"
if save_plots:
    output_dir.mkdir(parents=True, exist_ok=True)


# ========================================
# 1. Order book scenarios
# ========================================

baseline_depth = 250_000_000
baseline_width = 0.01
baseline_spread = 0.005 / 100
baseline_k = 0.5
baseline_resilience = pd.Timedelta("5min")

baseline_orderbook = Orderbook(
    name="baseline",
    depth=baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

depth_orderbook = Orderbook(
    name="depth 0_5x",
    depth=0.5 * baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

width_orderbook = Orderbook(
    name="width 2x",
    depth=baseline_depth,
    width=2 * baseline_width,
    spread=baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

spread_orderbook = Orderbook(
    name="spread 2x",
    depth=baseline_depth,
    width=baseline_width,
    spread=2 * baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

shape1_orderbook = Orderbook(
    name="shape k1",
    depth=baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=1,
    resilience=baseline_resilience,
)

shape2_orderbook = Orderbook(
    name="shape k1_5",
    depth=baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=1.5,
    resilience=baseline_resilience,
)

# Half-life does not change a static book, so it has no plot of its own
orderbooks_list = [
    baseline_orderbook,
    depth_orderbook,
    width_orderbook,
    spread_orderbook,
    shape1_orderbook,
    shape2_orderbook,
]

for orderbook in orderbooks_list:
    fig = plot_orderbooks(
        orderbook,
        show_parameters=False, show_legend=False,
        x_limit=2.2, y_limit=300_000_000,
        figsize=(5.5, 3.1),  # small canvas = large text relative to the plot (saved at 300 dpi)
        baseline=baseline_orderbook,  # dotted outline; skipped on the baseline's own plot
        title="",
        )
    fig.show()

    if save_plots:
        fig.savefig(output_dir / f"orderbook_{orderbook.name}.png", dpi=300, bbox_inches="tight")


# ========================================
# 2. Simulation outputs
# ========================================

experiment = "1s_2kappas"  # folder in data/outputs/GBM_simulations/ written by simulation_from_gbm.py
single_path_vol = "vol_100"  # resets are too rare at lower volatilities to show one path
single_path_seed = 579  # Cherry-picked to my hearts content for illustrative purposes

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

results_dir = GBM_SIMULATIONS / experiment / "iterations"
paths = sorted(results_dir.glob("seed_*.parquet"))
if not paths:
    raise FileNotFoundError(f"No saved iterations in {results_dir}; run simulation_from_gbm.py first")
runs = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
runs["vol"] = runs["price_series"].str.removeprefix("vol_").astype(int)
runs["active"] = runs["n_rebalances"] > 0  # paths with at least one reset
print(f"{runs['seed'].nunique()} iterations from {results_dir}")

# ---------- numbers for the slides ----------
def save_latex_table(table: pd.DataFrame, name: str, formats: dict, column_format: str) -> None:
    """
    Save a table as a bare booktabs tabular, to \\input into the deck (needs \\usepackage{booktabs}).
    Headers and row labels are written as given, so they may hold LaTeX such as $\\times$.
    """
    # Row labels become the first column, so the index name sits in the header row, not under it
    latex = table.reset_index().style.hide(axis="index").format(formats, na_rep="--").to_latex(
        column_format=column_format, hrules=True
    )
    if save_plots:
        (output_dir / f"{name}.tex").write_text(latex, encoding="utf-8")


pd.set_option("display.width", 250, "display.max_columns", 50)
active = runs[runs["orderbook"] == "baseline"].groupby("vol")["active"].agg(["sum", "mean"])
print("\nPaths with at least one reset (baseline book):")
print(active)
active_tex = pd.DataFrame(
    {"Paths with a reset": active["sum"], "Share of paths": active["mean"] * 100},
)
active_tex.index = [f"{vol}\\%" for vol in active_tex.index]
active_tex.index.name = "Volatility"
save_latex_table(active_tex, "active_paths", {"Paths with a reset": "{:,.0f}", "Share of paths": "{:.1f}\\%"}, "lrr")

# Trade size and impact are NaN on paths without resets, so their means are over the active paths
table = runs[runs["price_series"] == single_path_vol].groupby("orderbook", sort=False).agg(
    resets=("n_rebalances", "mean"),
    trade_k=("avg_rebalance_size", lambda s: s.mean() / 1e3),
    mean_abs_bps=("avg_abs_impact", lambda s: s.mean() * 1e4),
    max_buy_bps=("max_positive_impact", lambda s: s.mean() * 1e4),
    final_bps=("final_price_deviation", lambda s: s.mean() * 1e4),
    highest_bps=("max_price_deviation", lambda s: s.mean() * 1e4),
    largest_pct=("final_price_deviation", lambda s: s.max() * 1e2),
    nav_dir_bn=("final_nav_GBM_UP", lambda s: s.mean() / 1e9),
    nav_inv_m=("final_nav_GBM_DOWN", lambda s: s.mean() / 1e6),
)
print(f"\nSummary table at {single_path_vol}:\n{table.round(3)}")
# Slide version of the table: a tabular* to \input inside the frame, which sets the font size and
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
    "resets": "{:.1f}", "trade_k": "{:,.0f}", "mean_abs_bps": "{:.2f}", "max_buy_bps": "{:.2f}",
    "final_bps": "{:.1f}", "highest_bps": "{:.1f}", "largest_pct": "{:.2f}",
}
summary_lines = [
    r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}",
    r"    \toprule",
    r"    & & \textbf{Trade} & \multicolumn{2}{c}{\textbf{Impact (bps)}} & \multicolumn{3}{c}{\textbf{Spillover}}\\",
    r"    \cmidrule(lr){4-5}\cmidrule(lr){6-8}",
    r"    \textbf{Order book} & \textbf{Resets} & (\$k) & Mean $|I|$ & Max buy"
    r" & Final (bps) & Highest (bps) & Largest (\%)\\",
    r"    \midrule",
]
for book, label in summary_rows.items():
    cells = [label]
    for column, fmt in summary_columns.items():
        value = table.loc[book, column]
        # Without an order book there is no impact to report
        no_impact = book == "no_orderbook" and column in ("mean_abs_bps", "max_buy_bps")
        cells.append("--" if no_impact or pd.isna(value) else fmt.format(value))
    if book in highlighted_rows:
        cells = [rf"\red{{{cell}}}" for cell in cells]
    summary_lines.append("    " + " & ".join(cells) + r"\\")
summary_lines += [r"    \bottomrule", r"\end{tabular*}", ""]
if save_plots:
    summary_path = output_dir / f"summary_table_{single_path_vol.replace('_', '')}.tex"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

# ---------- spillover and resets by volatility ----------
means = runs.groupby(["orderbook", "vol"]).agg(
    final=("final_price_deviation", "mean"), resets=("n_rebalances", "mean")
)
# 95% confidence intervals of the means, from the standard deviations across seeds and the
# number of runs per cell on the "stdev" sheet written by simulation_from_gbm_table.py.
# Normal approximation: half-width 1.96 * std / sqrt(n), with n = 1000 runs per cell.
stdevs = pd.read_excel(
    GBM_SIMULATIONS / experiment / "summary_table.xlsx", sheet_name="stdev", index_col=[0, 1]
).reset_index()
stdevs["vol"] = stdevs["price_series"].str.removeprefix("vol_").astype(int)
stdevs = stdevs.set_index(["orderbook", "vol"])
ci95 = stdevs[["final_price_deviation", "n_rebalances"]].mul(1.96 / np.sqrt(stdevs["n_runs"]), axis=0)
dodge = np.linspace(-2.4, 2.4, len(book_styles))  # vol-% shift per book, so the bars do not overlap
with plt.rc_context(output_style):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    for offset, (book, (label, color, ls, marker, lw)) in zip(dodge, book_styles.items()):
        m, ci = means.loc[book], ci95.loc[book].reindex(means.loc[book].index)
        style = dict(color=color, ls=ls, marker=marker, lw=lw, ms=6, label=label,
                     elinewidth=1.0, capsize=3, alpha=0.9)
        axes[0].errorbar(m.index + offset, m["final"] * 1e4, yerr=ci["final_price_deviation"] * 1e4, **style)
        axes[1].errorbar(m.index + offset, m["resets"], yerr=ci["n_rebalances"], **style)
    axes[0].set_title("Cumulative spillover after 40 hours", fontsize=15)
    axes[0].set_ylabel("Mean final spillover (bps)")
    axes[1].set_title("Number of resets", fontsize=15)
    axes[1].set_ylabel("Mean resets per path")
    for ax in axes:
        ax.set_xlabel("Annualised GBM volatility (%)")
        ax.set_xticks([20, 40, 60, 80, 100])
    axes[1].set_ylim(bottom=0)  # resets are counts; spillover keeps matplotlib's automatic limits
    axes[0].legend(fontsize=12, loc="upper left")
    fig.tight_layout()
    fig.show()
    if save_plots:
        fig.savefig(output_dir / "gbm_updown_summary.png", dpi=300, bbox_inches="tight")

# ---------- one path, re-simulated in full ----------
single_path_books = ["baseline", "depth 0.5x", "spread 2x"]
summary, results = simulate_seed(
    single_path_seed,
    **{
        **setup,
        "volatilities": [int(single_path_vol.removeprefix("vol_")) / 100],
        "orderbooks": [ob for ob in orderbooks if ob is not None and ob.name in single_path_books],
    },
    keep_results=True,
)
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

with plt.rc_context(output_style):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    ax = axes[0]
    ax.plot(hours[::step], base["raw_price"].iloc[::step], color=GREY, lw=1.2, label="GBM input (no impact)")
    ax.plot(hours[::step], base["simulated_price"].iloc[::step], color=BLACK, lw=1.2, label="Baseline book")
    ax.plot(hours[::step], res["spread 2x"]["simulated_price"].iloc[::step], color=VERMILLION, lw=1.2,
            label=r"Spread 2$\times$")
    ax.set_title("Input price and simulated price", fontsize=15)
    ax.set_ylabel("Price")
    ax.legend(fontsize=12, loc="upper left")
    ax = axes[1]
    for book, label, color, ls in [
        ("baseline", "Baseline book", BLACK, "-"),
        ("depth 0.5x", r"Depth 0.5$\times$", PINK, "--"),
        ("spread 2x", r"Spread 2$\times$", VERMILLION, "-"),
    ]:
        ax.plot(hours[::step], (res[book]["price_multiplier"].iloc[::step] - 1) * 100,
                color=color, lw=1.6, ls=ls, label=label)
    ax.axhline(0, color=GREY, lw=0.8)  # grey, so it is not mistaken for the black baseline
    ax.set_title("Cumulative spillover from rebalancing", fontsize=15)
    ax.set_ylabel("Deviation from input price (%)")
    ax.legend(fontsize=12, loc="upper left")
    for ax in axes:
        ax.set_xlabel("Time elapsed (hours)")
        ax.set_xticks(range(0, 41, 5))
    fig.tight_layout()
    fig.show()
    if save_plots:
        fig.savefig(output_dir / f"{file_stem}_price.png", dpi=300, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.0), sharey=True)
    for ax, product, name, color in [
        (axes[0], gbm_up, "Direct product", NAVY),
        (axes[1], gbm_down, "Inverse product", RED),
    ]:
        delta = base[f"actual_delta_{product.name}"]
        n_resets = int((delta != 0).sum())
        ax.plot(hours[::step], base[f"leverage_after_{product.name}"].abs().iloc[::step], color=color, lw=1.2)
        for bound in (abs(product.leverage_lower_bound), abs(product.leverage_upper_bound)):
            ax.axhline(bound, color="black", lw=1.0, ls="--")
        ax.set_title(f"{name}: {n_resets} reset{'' if n_resets == 1 else 's'}", fontsize=15)
        ax.set_xlabel("Time elapsed (hours)")
        ax.set_xticks(range(0, 41, 5))
        ax.set_ylim(1.05, 3.2)
        print(f"{name}: {n_resets} resets, {int((delta > 0).sum())} purchases, {int((delta < 0).sum())} sales")
    axes[0].set_ylabel(r"Leverage after any reset, $|\lambda_t^*|$")
    fig.tight_layout()
    fig.show()
    if save_plots:
        fig.savefig(output_dir / f"{file_stem}_leverage.png", dpi=300, bbox_inches="tight")
