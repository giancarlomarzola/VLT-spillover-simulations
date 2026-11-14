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
single_path_seed = 88  # middle path, by number of resets, of those with any reset at vol_100

NAVY, RED, PURPLE, GREY = "#142D50", "#990000", "#7B4FA3", "#808080"
output_style = {
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
}
# label, colour, line style, marker, line width
book_styles = {
    "baseline": ("Baseline", NAVY, "-", "o", 2.6),
    "resilience 30min": ("Half-life 30 min", "#4A6583", "--", "s", 1.8),
    "depth 0.5x": (r"Depth 0.5$\times$", PURPLE, "--", "v", 1.8),
    "width 2x": (r"Width 2$\times$", "#C68A2E", "-", "D", 1.8),
    "shape k1": (r"Shape $\kappa=1$", "#8FA3B8", "-", "^", 1.8),
    "shape k1_5": (r"Shape $\kappa=1.5$", "#3E8E7E", ":", "P", 1.8),
    "spread 2x": (r"Spread 2$\times$", RED, "-", "o", 2.6),
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
pd.set_option("display.width", 250, "display.max_columns", 50)
print("\nPaths with at least one reset (baseline book):")
print(runs[runs["orderbook"] == "baseline"].groupby("vol")["active"].agg(["sum", "mean"]))

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

# ---------- spillover and resets by volatility ----------
means = runs.groupby(["orderbook", "vol"]).agg(
    final=("final_price_deviation", "mean"), resets=("n_rebalances", "mean")
)
with plt.rc_context(output_style):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    for book, (label, color, ls, marker, lw) in book_styles.items():
        m = means.loc[book]
        style = dict(color=color, ls=ls, marker=marker, lw=lw, ms=6, label=label)
        axes[0].plot(m.index, m["final"] * 1e4, **style)
        axes[1].plot(m.index, m["resets"], **style)
    axes[0].set_title("Cumulative spillover after 40 hours", fontsize=15)
    axes[0].set_ylabel("Mean final spillover (bps)")
    axes[1].set_title("Number of resets", fontsize=15)
    axes[1].set_ylabel("Mean resets per path")
    for ax in axes:
        ax.set_xlabel("Annualised GBM volatility (%)")
        ax.set_xticks([20, 40, 60, 80, 100])
        ax.set_ylim(bottom=0)
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
    ax.plot(hours[::step], base["simulated_price"].iloc[::step], color=NAVY, lw=1.2, label="Baseline book")
    ax.plot(hours[::step], res["spread 2x"]["simulated_price"].iloc[::step], color=RED, lw=1.2,
            label=r"Spread 2$\times$")
    ax.set_title("Input price and simulated price", fontsize=15)
    ax.set_ylabel("Price")
    ax.legend(fontsize=12, loc="upper left")
    ax = axes[1]
    for book, label, color, ls in [
        ("baseline", "Baseline book", NAVY, "-"),
        ("depth 0.5x", r"Depth 0.5$\times$", PURPLE, "--"),
        ("spread 2x", r"Spread 2$\times$", RED, "-"),
    ]:
        ax.plot(hours[::step], (res[book]["price_multiplier"].iloc[::step] - 1) * 100,
                color=color, lw=1.6, ls=ls, label=label)
    ax.axhline(0, color="black", lw=0.8)
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
