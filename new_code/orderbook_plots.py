from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from price_spillover_simulations import _as_side_pair

FIGURES_DIR = Path(__file__).resolve().parent.parent / "Figures"

# Orderbook formula
orderbook_formula = "linear"  # "linear" or "curved"
k = 0.5  # curvature parameter (scalar or (bid, ask) pair); only used if orderbook_formula == "curved"

# Orderbooks to plot (depth in USD, width/spread in %; each value can be a
# scalar for a symmetric book or a (bid, ask) pair for an asymmetric one).
# Matches the dashboard's default orderbook settings (.dashboard_config/dashboard_defaults.json).
orderbooks = {
    "Deep Narrow Tight"    : {"depth": 50_000_000, "width": 0.5, "spread": 0.005},
    "Deep Narrow Broad"    : {"depth": 50_000_000, "width": 0.5, "spread": 0.05},
    "Deep Wide Tight"      : {"depth": 50_000_000, "width": 2.0, "spread": 0.005},
    "Deep Wide Broad"      : {"depth": 50_000_000, "width": 2.0, "spread": 0.05},
    "Shallow Narrow Tight" : {"depth": 10_000_000, "width": 0.5, "spread": 0.005},
    "Shallow Wide Tight"   : {"depth": 10_000_000, "width": 2.0, "spread": 0.005},
    "Shallow Wide Broad"   : {"depth": 10_000_000, "width": 2.0, "spread": 0.05},
    "Asymmetrical"         : {"depth": (30_000_000, 50_000_000), "width": 0.5, "spread": 0.005},
}


def orderbook_depth_linear(x, D, W, S):
    """Piecewise-linear synthetic orderbook. D/W/S are each (bid, ask) pairs."""
    D_bid, D_ask = D
    W_bid, W_ask = W
    S_bid, S_ask = S
    return np.where(
        x <= -W_bid, D_bid,
        np.where(x < -S_bid, D_bid / (W_bid - S_bid) * (-x - S_bid),
        np.where(x < S_ask, 0,
        np.where(x < W_ask, D_ask / (W_ask - S_ask) * (x - S_ask), D_ask))))


def orderbook_depth_curved(x, D, W, S, k):
    """Curved synthetic orderbook f(z) = D * t/(k + (1-k)*t). D/W/S/k are each (bid, ask) pairs."""
    D_bid, D_ask = D
    W_bid, W_ask = W
    S_bid, S_ask = S
    k_bid, k_ask = k

    t_bid = (-x - S_bid) / (W_bid - S_bid)
    bid_curve = D_bid * t_bid / (k_bid + (1 - k_bid) * t_bid)

    t_ask = (x - S_ask) / (W_ask - S_ask)
    ask_curve = D_ask * t_ask / (k_ask + (1 - k_ask) * t_ask)

    return np.where(
        x <= -W_bid, D_bid,
        np.where(x <= -S_bid, bid_curve,
        np.where(x < S_ask, 0,
        np.where(x < W_ask, ask_curve, D_ask))))


def _format_side(pair):
    bid, ask = pair
    return f"{bid:g}" if bid == ask else f"{bid:g}/{ask:g}"


plot_orderbooks = {name: cfg for name, cfg in orderbooks.items() if cfg is not None}

n = len(plot_orderbooks)
ncols = 2
nrows = -(-n // ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 12.5))
axes = axes.flatten()

max_width = max(max(_as_side_pair(cfg["width"])) for cfg in plot_orderbooks.values())
x = np.linspace(-1.2 * max_width, 1.2 * max_width, 1000)

for ax, (name, cfg) in zip(axes, plot_orderbooks.items()):
    D = _as_side_pair(cfg["depth"])
    W = _as_side_pair(cfg["width"])
    S = _as_side_pair(cfg["spread"])

    if orderbook_formula == "curved":
        K = _as_side_pair(k)
        y = orderbook_depth_curved(x, D, W, S, K)
    else:
        y = orderbook_depth_linear(x, D, W, S)

    ax.fill_between(x, y, where=(x < 0), color="green", alpha=0.3, label="Cumulative Bid")
    ax.fill_between(x, y, where=(x > 0), color="red", alpha=0.3, label="Cumulative Ask")
    ax.plot(x[x <= 0], y[x <= 0], color="green", lw=0.7)
    ax.plot(x[x >= 0], y[x >= 0], color="red", lw=0.7)
    ax.set_title(f"{name}\nDepth=${_format_side(tuple(d / 1_000_000 for d in D))}M, "
                 f"Width={_format_side(W)}%, Spread={_format_side(S)}%")
    ax.set_xlabel("Price deviation (%)")
    ax.set_xlim(-1.2 * max_width, 1.2 * max_width)
    ax.set_ylabel("Depth (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(True, alpha=0.3)

for ax in axes[n:]:
    ax.axis("off")

formula_note = f"{orderbook_formula.capitalize()} orderbook" + (f" (k={k})" if orderbook_formula == "curved" else "")
fig.suptitle(f"Simulated Orderbook Archetypes — {formula_note}", fontsize=14, y=0.98)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95), fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.94])

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / f"orderbook_archetypes_{orderbook_formula}.png", dpi=300, bbox_inches="tight")
plt.show()
