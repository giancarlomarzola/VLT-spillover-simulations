"""Plots for the QuantMinds presentation."""

import matplotlib.pyplot as plt

from core.orderbook import Orderbook
from utils.paths import FIGURES_DIR
from utils.plotting import plot_orderbooks

save_plots = False

# Example orderbook (placeholder numbers)
example_orderbook = Orderbook(
    name="Example Orderbook",
    depth=50_000_000,
    width=0.01,
    spread=0.0005,
    k=0.3,
)

fig = plot_orderbooks(example_orderbook)

if save_plots:
    fig.savefig(FIGURES_DIR / "quant_minds_example_orderbook.png", dpi=300, bbox_inches="tight")
plt.show()
