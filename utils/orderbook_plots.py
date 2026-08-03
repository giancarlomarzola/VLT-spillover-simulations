import matplotlib.pyplot as plt
import numpy as np

from core.orderbook import Orderbook
from utils.paths import FIGURES_DIR

# Orderbook formula
k = 0.3 # for linear orderbooks = 1
deep, shallow = 50_000_000, 10_000_000
narrow, wide = 0.005, 0.01
tight, broad = 0.00005, 0.0005

save_plots = False

# Orderbooks to plot
plot_orderbooks = {
    "Deep Narrow Tight": Orderbook(
        name="Deep Narrow Tight",
        depth_bid=deep,
        depth_ask=deep,
        width_bid=narrow,
        width_ask=narrow,
        spread_bid=tight,
        spread_ask=tight,
        k_bid=k,
        k_ask=k,
    ),
    "Deep Narrow Broad": Orderbook(
        name="Deep Narrow Broad",
        depth_bid=deep,
        depth_ask=deep,
        width_bid=narrow,
        width_ask=narrow,
        spread_bid=broad,
        spread_ask=broad,
        k_bid=k,
        k_ask=k,
    ),
    "Deep Wide Tight": Orderbook(
        name="Deep Wide Tight",
        depth_bid=deep,
        depth_ask=deep,
        width_bid=wide,
        width_ask=wide,
        spread_bid=tight,
        spread_ask=tight,
        k_bid=k,
        k_ask=k,
    ),
    "Deep Wide Broad": Orderbook(
        name="Deep Wide Broad",
        depth_bid=deep,
        depth_ask=deep,
        width_bid=wide,
        width_ask=wide,
        spread_bid=broad,
        spread_ask=broad,
        k_bid=k,
        k_ask=k,
    ),
    "Shallow Narrow Tight": Orderbook(
        name="Shallow Narrow Tight",
        depth_bid=shallow,
        depth_ask=shallow,
        width_bid=narrow,
        width_ask=narrow,
        spread_bid=tight,
        spread_ask=tight,
        k_bid=k,
        k_ask=k,
    ),
    "Shallow Wide Tight": Orderbook(
        name="Shallow Wide Tight",
        depth_bid=shallow,
        depth_ask=shallow,
        width_bid=wide,
        width_ask=wide,
        spread_bid=tight,
        spread_ask=tight,
        k_bid=k,
        k_ask=k,
    ),
    "Shallow Wide Broad": Orderbook(
        name="Shallow Wide Broad",
        depth_bid=shallow,
        depth_ask=shallow,
        width_bid=wide,
        width_ask=wide,
        spread_bid=broad,
        spread_ask=broad,
        k_bid=k,
        k_ask=k,
    ),
    "Asymmetrical": Orderbook(
        name="Asymmetrical",
        depth_bid=deep/2,
        depth_ask=deep,
        width_bid=narrow,
        width_ask=narrow,
        spread_bid=tight,
        spread_ask=tight,
        k_bid=k,
        k_ask=k,
    ),
}


def orderbook_depth_curve(x, ob):
    """
    Cumulative depth f(z) = D * t/(k + (1-k)*t) at price deviation x (%), 
    from an Orderbook's parameters.
    """
    D_bid, D_ask = ob.depth_bid, ob.depth_ask
    W_bid, W_ask = ob.width_bid * 100, ob.width_ask * 100
    S_bid, S_ask = ob.spread_bid * 100, ob.spread_ask * 100
    k_bid, k_ask = ob.k_bid, ob.k_ask

    t_bid = (-x - S_bid) / (W_bid - S_bid)
    bid_curve = D_bid * t_bid / (k_bid + (1 - k_bid) * t_bid)

    t_ask = (x - S_ask) / (W_ask - S_ask)
    ask_curve = D_ask * t_ask / (k_ask + (1 - k_ask) * t_ask)

    return np.where(
        x <= -W_bid,
        D_bid,
        np.where(
            x <= -S_bid,
            bid_curve,
            np.where(x < S_ask, 0, np.where(x < W_ask, ask_curve, D_ask)),
        ),
    )


def _format_side(bid, ask):
    return f"{bid:g}" if bid == ask else f"{bid:g}/{ask:g}"


n = len(plot_orderbooks)
ncols = 2
nrows = -(-n // ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 12.5))
axes = axes.flatten()

max_width = wide * 100
x = np.linspace(-1.2 * max_width, 1.2 * max_width, 1000)

for ax, (name, ob) in zip(axes, plot_orderbooks.items()):
    y = orderbook_depth_curve(x, ob)

    ax.fill_between(
        x, y, where=(x < 0), color="green", alpha=0.3, label="Cumulative Bid"
    )
    ax.fill_between(x, y, where=(x > 0), color="red", alpha=0.3, label="Cumulative Ask")
    ax.plot(x[x <= 0], y[x <= 0], color="green", lw=0.7)
    ax.plot(x[x >= 0], y[x >= 0], color="red", lw=0.7)
    ax.set_title(
        f"{name}\nDepth=${_format_side(ob.depth_bid / 1_000_000, ob.depth_ask / 1_000_000)}M, "  # noqa: E501
        f"Width={_format_side(ob.width_bid * 100, ob.width_ask * 100)}%, "
        f"Spread={_format_side(ob.spread_bid * 100, ob.spread_ask * 100)}%"
    )
    ax.set_xlabel("Price deviation (%)")
    ax.set_xlim(-1.2 * max_width, 1.2 * max_width)
    ax.set_ylabel("Depth (USD)")
    ax.set_ylim(0, 1.2*deep)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(True, alpha=0.3)

for ax in axes[n:]:
    ax.axis("off")

orderbook_formula = "linear" if k == 1 else "concave" if k > 1 else "convex"
formula_note = f"{orderbook_formula.capitalize()} orderbook" + (
    f" (k={k})" if orderbook_formula != "linear" else ""
)
fig.suptitle(f"Simulated Orderbook Archetypes — {formula_note}", fontsize=14, y=0.98)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95), fontsize=10
)
plt.tight_layout(rect=[0, 0, 1, 0.94])

if save_plots:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        FIGURES_DIR / f"orderbook_archetypes_{orderbook_formula}.png",
        dpi=300,
        bbox_inches="tight",
    )
plt.show()
