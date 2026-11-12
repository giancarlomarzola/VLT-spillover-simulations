"""Plots for the QuantMinds presentation."""

from core.orderbook import Orderbook
from utils.paths import FIGURES_DIR
from utils.plotting import plot_orderbooks

save_plots = True


# ========================================
# Orderbook parameter examples
# ========================================

baseline_depth = 250_000_000
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
        depth=0.5 * baseline_depth,
        width=baseline_width,
        spread=baseline_spread,
        k=baseline_k,
        resilience=baseline_resilience,
    ),
    Orderbook(
        name="Smaller Width",
        depth=baseline_depth,
        width=0.5 * baseline_width,
        spread=baseline_spread,
        k=baseline_k,
        resilience=baseline_resilience,
    ),
    Orderbook(
        name="Larger Spread",
        depth=baseline_depth,
        width=baseline_width,
        spread=20 * baseline_spread,
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


baseline_orderbook = orderbooks_list[0]
output_dir = FIGURES_DIR / "quant_minds"
if save_plots:
    output_dir.mkdir(parents=True, exist_ok=True)

for orderbook in orderbooks_list:
    fig = plot_orderbooks(
        orderbook,
        show_parameters=False, show_legend=False,
        x_limit=1.2, y_limit=300_000_000,
        figsize=(4.8, 3.1),  # small canvas = large text relative to the plot (saved at 300 dpi)
        baseline=baseline_orderbook,  # dotted outline; skipped on the baseline's own plot
        )

    if save_plots:
        fig.savefig(output_dir / f"orderbook_{orderbook.name}.png", dpi=300, bbox_inches="tight")

