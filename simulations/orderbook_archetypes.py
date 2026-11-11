import matplotlib.pyplot as plt

from core.orderbook import Orderbook
from utils.paths import FIGURES_DIR
from utils.plotting import plot_orderbooks

# Orderbook formula
k = 0.3  # for linear orderbooks = 1
deep, shallow = 50_000_000, 10_000_000
narrow, wide = 0.005, 0.01
tight, broad = 0.00005, 0.0005

save_plots = False

# Orderbooks to plot
orderbooks = [
    Orderbook(name="Deep Narrow Tight", depth=deep, width=narrow, spread=tight, k=k),
    Orderbook(name="Deep Narrow Broad", depth=deep, width=narrow, spread=broad, k=k),
    Orderbook(name="Deep Wide Tight", depth=deep, width=wide, spread=tight, k=k),
    Orderbook(name="Deep Wide Broad", depth=deep, width=wide, spread=broad, k=k),
    Orderbook(name="Shallow Narrow Tight", depth=shallow, width=narrow, spread=tight, k=k),
    Orderbook(name="Shallow Wide Tight", depth=shallow, width=wide, spread=tight, k=k),
    Orderbook(name="Shallow Wide Broad", depth=shallow, width=wide, spread=broad, k=k),
    Orderbook(name="Asymmetrical", depth=(deep / 2, deep), width=narrow, spread=tight, k=k),
]

orderbook_formula = "linear" if k == 1 else "concave" if k > 1 else "convex"
formula_note = f"{orderbook_formula.capitalize()} orderbook" + (f" (k={k})" if orderbook_formula != "linear" else "")

fig = plot_orderbooks(orderbooks, cols=2, title=f"Simulated Orderbook Archetypes — {formula_note}")

if save_plots:
    fig.savefig(FIGURES_DIR / f"orderbook_archetypes_{orderbook_formula}.png", dpi=300, bbox_inches="tight")
plt.show()
