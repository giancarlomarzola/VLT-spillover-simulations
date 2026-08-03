import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from utils.paths import DATA_PROCESSED
from utils.plotting import TRACE_COLORS

currency = "btc"
frequency = "50ms"

df = pd.read_parquet(DATA_PROCESSED / f"{currency}_{frequency}_processed.parquet")

df["log_return"] = np.log(df["price"] / df["price"].shift(1))

# Log returns
print("Summary statistics for log returns (in basis points)")
log_returns_bp = (df["log_return"] * 100).copy()
display(log_returns_bp.describe().apply("{0:,.5f}".format))
print(f"1st quantile:\t {log_returns_bp.quantile(0.01)}")
print(f"99th quantile:\t  {log_returns_bp.quantile(0.99)}")


# Get vector with only shock returns (= returns > some cutoff e.g. i-th percentile)
# Maybe for future: Get cutoff value from stable series and apply to crash series

eta_lower = log_returns_bp.quantile(0.01)
eta_upper = log_returns_bp.quantile(0.99)

df["log_shocks"] = log_returns_bp.where((log_returns_bp < eta_lower) | (log_returns_bp > eta_upper), 0)

print("\n\nseries of log_returns that exceed eta:")
display(df["log_shocks"][df["log_shocks"] != 0])

# Scatter plot of log returns, highlighting shocks (outside eta_lower/eta_upper) in red
is_shock = df["log_shocks"] != 0

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(
    log_returns_bp.index[~is_shock],
    log_returns_bp[~is_shock],
    color=TRACE_COLORS[0],
    s=1,
    label=f"Normal (n={(~is_shock).sum():,})",
    alpha=0.8,
)
ax.scatter(
    log_returns_bp.index[is_shock],
    log_returns_bp[is_shock],
    color=TRACE_COLORS[1],
    s=1,
    label=f"Shock (n={is_shock.sum():,})",
    alpha=0.8,
)

ax.axhline(eta_upper, color="gray", linewidth=1, linestyle="dotted")
ax.axhline(eta_lower, color="gray", linewidth=1, linestyle="dotted")
ax.text(
    log_returns_bp.index[0],
    eta_upper,
    f"eta_upper = {eta_upper:.4f}",
    va="bottom",
    color="gray",
)
ax.text(
    log_returns_bp.index[0],
    eta_lower,
    f"eta_lower = {eta_lower:.4f}",
    va="top",
    color="gray",
)

ax.set_ylim(5 * eta_lower, 5 * eta_upper)
ax.set_title("Log Returns (bp) with Shock Thresholds")
ax.set_xlabel("Index")
ax.set_ylabel("Log Return (bp)")
ax.legend()

plt.show()
