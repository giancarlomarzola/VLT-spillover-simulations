"""Plots for the QuantMinds presentation."""

import pandas as pd

from core.orderbook import Orderbook
from utils.paths import FIGURES_DIR
from utils.plotting import plot_orderbooks

save_plots = False


# ========================================
# Orderbook parameter examples
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
    name="depth 0.5x",
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

# None = no orderbook: rebalances have no price impact (the reference run)
orderbooks_list = [
    baseline_orderbook,
    depth_orderbook,
    width_orderbook,
    spread_orderbook,
    shape1_orderbook,
    shape2_orderbook,
]


baseline_orderbook = orderbooks_list[0]
output_dir = FIGURES_DIR / "quant_minds"
if save_plots:
    output_dir.mkdir(parents=True, exist_ok=True)

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

