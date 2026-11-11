"""Plots for the QuantMinds presentation."""

import importlib  # noqa: F401 (interactive use) importlib.reload(nameOfModule)

import matplotlib.pyplot as plt

from core.orderbook import Orderbook
from utils.paths import FIGURES_DIR
from utils.plotting import plot_orderbooks

save_plots = False

baseline_depth = 50_000_000
baseline_width = 0.01
baseline_spread = 0.005 / 100
baseline_k = 0.3
baseline_resilience = 0 

baseline_orderbook = Orderbook(
    name="baseline",
    depth=baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

depth_orderbook = Orderbook(
    name="depth",
    depth=0.5 * baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

width_orderbook = Orderbook(
    name="width",
    depth=baseline_depth,
    width=0.5 * baseline_width,
    spread=baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

spread_orderbook = Orderbook(
    name="spread",
    depth=baseline_depth,
    width=baseline_width,
    spread=2 * baseline_spread,
    k=baseline_k,
    resilience=baseline_resilience,
)

shape1_orderbook = Orderbook(
    name="shape1",
    depth=baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=3,
    resilience=baseline_resilience,
)

shape2_orderbook = Orderbook(
    name="shape2",
    depth=baseline_depth,
    width=baseline_width,
    spread=baseline_spread,
    k=1,
    resilience=baseline_resilience,
)


orderbooks_list = [
    baseline_orderbook,
    depth_orderbook,
    width_orderbook,
    spread_orderbook,
    shape1_orderbook,
    shape2_orderbook,
    ]

for orderbook in orderbooks_list:
    fig = plot_orderbooks(orderbook, show_parameters=False, x_limit=1.2, y_limit=60_000_000)

    if save_plots:
        fig.savefig(FIGURES_DIR / f"quant_minds_orderbook_{orderbook.name}.png", dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()