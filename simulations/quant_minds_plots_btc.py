"""
Figures for the QuantMinds presentation, saved to data/outputs/figures/quant_minds/.

1. Order book illustration: the baseline book, with each parameter labelled.
2. Simulation outputs, from the BTC bootstrap experiment saved by run_btc_bootstrap_experiment.py:
     btc_seed<seed>_price.png / _leverage.png / _nav.png: one path, re-simulated in full here.
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

save_plots = True

output_dir = FIGURES_DIR / "quant_minds"
if save_plots:
    output_dir.mkdir(parents=True, exist_ok=True)


# ========================================
# 1. Order book illustration
# ========================================

# The baseline BTC book of the bootstrap experiment (simulations.products_and_orderbooks), with each
# parameter labelled. The scenarios (one parameter changed at a time) are listed in a table on the
# slides instead of plotted. The ask side shows the full book; the bid side is drawn after a sale has
# used up part of its depth, to illustrate resilience.
baseline_orderbook = next(ob for ob in btc_orderbooks() if ob is not None and ob.name == "baseline")
d = baseline_orderbook.depth_ask_full / 1e6  # US$m
w = baseline_orderbook.width_ask * 100  # %
c = baseline_orderbook.spread_ask * 100  # %
baseline_kappa = baseline_orderbook.k_ask
half_life_min = baseline_orderbook.resilience_ask / 60
consumed = 0.4  # share of the bid depth used up in the illustration
d_left = (1 - consumed) * d

ob_bid_col, ob_ask_col = (153 / 255, 0, 0), (20 / 255, 45 / 255, 80 / 255)
highlight_col = "#990000"  # the deck's red, for the parameter a scenario changes


def cumulative_depth(z, depth, kappa):
    """Cumulative depth f(z) = depth * tau / (kappa + (1 - kappa) tau) at distance z (%) from mid."""
    tau = np.clip((np.abs(z) - c) / (w - c), 0, 1)
    return depth * tau / (kappa + (1 - kappa) * tau)


def plot_orderbook_illustration(kappa, file_name):
    """
    The baseline book with shape kappa, every parameter labelled. With a kappa other than the
    baseline's, the shape label is in red and the baseline ask curve is drawn dotted for reference.
    """
    x_max, y_max = 2.2, 79
    x_ask = np.linspace(0, x_max, 2000)
    x_bid = -x_ask
    arrow = dict(arrowstyle="<->", color="black", lw=1.5, shrinkA=0, shrinkB=0)
    fs = 22  # label font size
    changed = kappa != baseline_kappa

    with plt.rc_context({"mathtext.fontset": "cm"}):
        fig = plt.figure(figsize=(12, 6.2))
        ax = fig.add_axes([0.09, 0.14, 0.64, 0.84])  # main plot; the half-spread zoom goes to its right
        # Ask side: the full book
        ax.fill_between(x_ask, cumulative_depth(x_ask, d, kappa), color=ob_ask_col, alpha=0.8, lw=0)
        ax.plot(x_ask, cumulative_depth(x_ask, d, kappa), color=ob_ask_col, lw=0.8)
        if changed:
            ax.plot(x_ask, cumulative_depth(x_ask, d, baseline_kappa), color="black", lw=1.4, ls=":")
        # Bid side: depth left after a sale, and the consumed depth (hatched) that refills over time
        ax.fill_between(x_bid, cumulative_depth(x_bid, d_left, kappa), color=ob_bid_col, alpha=0.8, lw=0)
        ax.fill_between(x_bid, cumulative_depth(x_bid, d_left, kappa), cumulative_depth(x_bid, d, kappa),
                        facecolor="none", edgecolor=ob_bid_col, hatch="///", lw=0, alpha=0.5)
        ax.plot(x_bid, cumulative_depth(x_bid, d, kappa), color=ob_bid_col, lw=0.8, ls=":")
        ax.plot(x_bid, cumulative_depth(x_bid, d_left, kappa), color=ob_bid_col, lw=0.8)
        ax.text(-1.75, 9, "Bid", color="white", fontsize=fs + 1, ha="center", va="center", weight="bold")
        ax.text(1.75, 9, "Ask", color="white", fontsize=fs + 1, ha="center", va="center", weight="bold")

        # Depth d: from zero to the plateau of the ask side
        ax.annotate("", xy=(2.1, 0), xytext=(2.1, d), arrowprops=arrow | {"color": "white"})
        ax.text(2.02, d / 2 + 6, f"Depth $d$\n= US\\${d:.0f}m", color="white", fontsize=fs, ha="right", va="center")

        # Half-width w: from the mid-quote to the last quote, on the bid side (clear of the shape label)
        ax.annotate("", xy=(0, d + 4), xytext=(-w, d + 4), arrowprops=arrow)
        ax.plot([-w, -w], [d, d + 6], color="black", lw=0.8, ls="--")
        ax.plot([0, 0], [0, d + 6], color="black", lw=0.8, ls="--")
        ax.text(-0.75, d + 6, f"Half-width $w$ = {w * 100:.0f} bps", fontsize=fs, ha="center", va="bottom")

        # Shape kappa: marked on the ask curve at z = 0.5%, halfway through the range, so the
        # baseline and kappa scenarios can be compared at the same point; the arrow drops straight to it
        z_mark = 0.5
        f_mark = float(cumulative_depth(np.array(z_mark), d, kappa))
        ax.plot([z_mark, z_mark], [0, f_mark], color="white", lw=0.8, ls="--")
        ax.plot(z_mark, f_mark, "o", color="white", ms=7, mec="black", mew=0.6, zorder=5)
        ax.annotate("", xy=(z_mark, f_mark + 1.5), xytext=(z_mark, 64),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5, shrinkA=0, shrinkB=0))
        ax.text(z_mark - 0.12, 65, f"Shape $\\kappa = {kappa:g}$", fontsize=fs, ha="left", va="bottom",
                color=highlight_col if changed else "black", weight="bold" if changed else "normal")

        # Half-life h: the consumed bid depth refills
        ax.annotate("", xy=(-1.6, d - 0.5), xytext=(-1.6, d_left + 0.5),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.8))
        ax.text(-2.15, d + 17, f"Consumed depth refills:\nhalf-life $h$ = {half_life_min:.0f} min",
                fontsize=fs, ha="left", va="bottom")

        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Price distance from mid-quote, $z$ (%)", fontsize=fs)
        ax.set_ylabel("Cumulative depth (US\\$m)", fontsize=fs)
        ax.tick_params(labelsize=fs - 2)
        ax.set_yticks(range(0, 61, 10))
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_bounds(0, 60)

        # Half-spread c: too small to see at this scale, so zoom in on the mid-quote, to the right
        inset = fig.add_axes([0.79, 0.3, 0.2, 0.32])
        zoom = 2.5 * c
        x_zoom = np.linspace(-zoom, zoom, 1001)
        pos, neg = x_zoom >= 0, x_zoom <= 0
        inset.fill_between(x_zoom[pos], cumulative_depth(x_zoom[pos], d, kappa), color=ob_ask_col, alpha=0.8, lw=0)
        inset.fill_between(x_zoom[neg], cumulative_depth(x_zoom[neg], d_left, kappa), color=ob_bid_col, alpha=0.8, lw=0)
        inset.set_xlim(-zoom, zoom)
        inset_top = cumulative_depth(np.array(zoom), d, kappa) * 1.05
        inset.set_ylim(0, inset_top)
        inset.set_xticks([-c, 0, c], ["$-c$", "$0$", "$c$"], fontsize=fs)
        inset.set_yticks([])
        inset.tick_params(length=3, pad=3)
        inset.axvline(0, color="black", lw=0.8, ls="--")
        inset.annotate("", xy=(0, 0.75 * inset_top), xytext=(c, 0.75 * inset_top), arrowprops=arrow)
        inset.set_title(f"Half-spread\n$c$ = {c * 100:g} bps", fontsize=fs, pad=8)

        if save_plots:
            fig.savefig(output_dir / file_name, dpi=300, bbox_inches="tight", pad_inches=0.02)
        else:
            fig.show()


plot_orderbook_illustration(baseline_kappa, "orderbook_illustration.png")
plot_orderbook_illustration(1.5, "orderbook_illustration_k1_5.png")  # the shape k1_5 scenario


# ========================================
# 2. Simulation outputs
# ========================================

# folder in data/outputs/BTC_bootstrap_simulations/ written by run_btc_bootstrap_experiment.py
experiment = "100ms_4h_block1s"  
# Cherry-picked for illustrative purposes. 841: price rises, only BTCDOWN resets (buys).
# 619: price falls, both products reset (both sell); the depth 0.5x book ends in a feedback loop.
# 262: price falls ~12%, both products reset (both sell), no feedback loop on any book.
# Override from the command line: python simulations/quant_minds_plots_btc.py 619
single_path_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 262

# Okabe-Ito palette: distinguishable under the common forms of colour blindness
BLACK, BLUE, VERMILLION = "#000000", "#0072B2", "#D55E00"
# Presentation colours: the leverage panels, and the Depth 0.5x line on the single-path plots
NAVY, RED = "#142D50", "#990000"
output_style = {
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
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
    if book in highlighted_rows:
        cells = [rf"\red{{{cell}}}" for cell in cells]
    summary_lines.append("    " + " & ".join(cells) + r"\\")
summary_lines += [r"    \bottomrule", r"\end{tabular}", ""]
if save_plots:
    (output_dir / "summary_table_btc.tex").write_text("\n".join(summary_lines), encoding="utf-8")

# ---------- one path, re-simulated in full ----------
single_path_books = ["baseline", "depth 0.5x", "spread 2x"]
setup = load_setup(experiment_dir)  # the setup the saved results were run with
btc_up, btc_down = setup["vletps"]
# Keep the no-orderbook run too (None): the NAV plot uses it as the no-impact reference
setup["orderbooks"] = [ob for ob in setup["orderbooks"] if ob is None or ob.name in single_path_books]
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
res = {book: results[price_label, book] for book in single_path_books + ["no_orderbook"]}
base = res["baseline"]
hours = base.index.total_seconds() / 3600
step = 10  # plot every 10th step (once a second): plenty for a 4-hour axis
file_stem = f"btc_seed{single_path_seed}"
hour_ticks = np.arange(0, 4.01, 0.5)
# Single-path line styles, in legend order: book, label, colour, line width, z-order. The input price
# (the no-orderbook run) is black and drawn on top; depth uses the deck's dark red.
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
    ax.set_ylabel("BTC price (US\\$)")
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

    # NAV of each product on the same path: no-impact reference and the three books
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    for ax, product, name in [(axes[0], btc_up, "Direct (BTCUP)"), (axes[1], btc_down, "Inverse (BTCDOWN)")]:
        for book, label, color, lw, zorder in path_styles:
            ax.plot(hours[::step], res[book][f"nav_{product.name}"].iloc[::step] / 1e6,
                    color=color, lw=lw, label=label, zorder=zorder)
        ax.set_title(name, fontsize=15)
        ax.set_xlabel("Time elapsed (hours)")
        ax.set_xticks(hour_ticks)
        print(f"{name} final NAV (US$m): " + ", ".join(
            f"{book} {res[book][f'nav_{product.name}'].iloc[-1] / 1e6:.2f}"
            for book in ["no_orderbook"] + single_path_books))
    axes[0].set_ylabel(r"NAV (US\$m)")
    axes[0].legend(fontsize=12, loc="best")
    fig.tight_layout()
    if save_plots:
        fig.savefig(output_dir / f"{file_stem}_nav.png", dpi=300, bbox_inches="tight")
    else:
        fig.show()
print(f"Seed {single_path_seed} final spillover: "
      + ", ".join(f"{book} {res[book]['price_multiplier'].iloc[-1] - 1:.2%}" for book in single_path_books))
