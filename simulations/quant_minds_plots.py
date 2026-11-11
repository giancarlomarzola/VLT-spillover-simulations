"""Plots for the QuantMinds presentation."""

from core.orderbook import Orderbook
from utils.paths import FIGURES_DIR
from utils.plotting import plot_orderbooks

save_plots = True


# ========================================
# Orderbook parameter examples
# ========================================

baseline_depth = 50_000_000
baseline_width = 0.01
baseline_spread = 0.005 / 100
baseline_k = 0.3
baseline_resilience = 0 

orderbooks_list = [
    Orderbook(
        name="Baseline",
        depth=baseline_depth,
        width=baseline_width,
        spread=baseline_spread,
        k=baseline_k,
        resilience=baseline_resilience,
    ),
    Orderbook(
        name="Lower Depth",
        depth=0.3 * baseline_depth,
        width=baseline_width,
        spread=baseline_spread,
        k=baseline_k,
        resilience=baseline_resilience,
    ),
    Orderbook(
        name="Larger Width",
        depth=baseline_depth,
        width=1.5 * baseline_width,
        spread=baseline_spread,
        k=baseline_k,
        resilience=baseline_resilience,
    ),
    Orderbook(
        name="Larger Spread",
        depth=baseline_depth,
        width=baseline_width,
        spread=10 * baseline_spread,
        k=baseline_k,
        resilience=baseline_resilience,
    ),
    Orderbook(
        name="Linear",
        depth=baseline_depth,
        width=baseline_width,
        spread=baseline_spread,
        k=1,
        resilience=baseline_resilience,
    ),
    Orderbook(
        name="Concave",
        depth=baseline_depth,
        width=baseline_width,
        spread=baseline_spread,
        k=5,
        resilience=baseline_resilience,
    ),
]


for orderbook in orderbooks_list:
    fig = plot_orderbooks(
        orderbook, 
        show_parameters=False, show_legend=False, 
        x_limit=1.2, y_limit=60_000_000
        )

    if save_plots:
        fig.savefig(FIGURES_DIR / "quant_minds" / f"orderbook_{orderbook.name}.png", dpi=300, bbox_inches="tight")

