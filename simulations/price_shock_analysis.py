import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from utils.data_io import processed_filename
from utils.paths import DATA_PROCESSED
from utils.plotting import TRACE_COLORS

currency = "btc"
frequency = "1s"

df = pd.read_parquet(DATA_PROCESSED / processed_filename(currency, frequency))

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

df["log_shocks_bp"] = log_returns_bp.where((log_returns_bp < eta_lower) | (log_returns_bp > eta_upper), 0)

print("\n\nseries of log_returns that exceed eta:")
display(df[["timestamp", "log_shocks_bp"]][df["log_shocks_bp"] != 0])

# Scatter plot of log returns, highlighting shocks (outside eta_lower/eta_upper) in red
is_shock = df["log_shocks_bp"] != 0

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(
    df["timestamp"][~is_shock],
    log_returns_bp[~is_shock],
    color=TRACE_COLORS[0],
    s=1,
    label=f"Normal (n={(~is_shock).sum():,})",
    alpha=0.8,
)
ax.scatter(
    df["timestamp"][is_shock],
    log_returns_bp[is_shock],
    color=TRACE_COLORS[1],
    s=1,
    label=f"Shock (n={is_shock.sum():,})",
    alpha=0.8,
)

ax.axhline(eta_upper, color="gray", linewidth=1, linestyle="dotted")
ax.axhline(eta_lower, color="gray", linewidth=1, linestyle="dotted")
ax.text(
    df["timestamp"].iloc[0],
    eta_upper,
    f"eta_upper = {eta_upper:.4f}",
    va="bottom",
    color="gray",
)
ax.text(
    df["timestamp"].iloc[0],
    eta_lower,
    f"eta_lower = {eta_lower:.4f}",
    va="top",
    color="gray",
)

ax.set_ylim(5 * eta_lower, 5 * eta_upper)
ax.set_title("Log Returns (bp) with Shock Thresholds")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Log Return (bp)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.legend()

plt.show()


# Load binance data for different time to apply shocks to

start_time = pd.Timestamp("2021-05-19 4:00:00", tz="UTC")
end_time = pd.Timestamp("2021-05-19 6:00:00", tz="UTC")

df_unshocked = pd.read_parquet(DATA_PROCESSED / processed_filename(currency, frequency, start_time, end_time))

shock_factor = np.exp((df["log_shocks_bp"] / 100).cumsum())
df_unshocked["price_shocked"] = df_unshocked["price"] * shock_factor.to_numpy()


# Plot prices from before crash, during crash, and before crash with artificial shocks
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    df_unshocked.index, df_unshocked["price"],
    color=TRACE_COLORS[0], linewidth=1,
    label="04:00 - 06:00 price"
    )
ax.plot(
    df.index, df["price"],
    color=TRACE_COLORS[1], linewidth=1,
    label="12:00 - 14:00 price"
    )
ax.plot(
    df_unshocked.index, df_unshocked["price_shocked"],
    color=TRACE_COLORS[2], linewidth=1,
    label="04:00 - 06:00 price (shocked)"
    )

ax.set_ylim(0, 1.2 * df["price"].max())
ax.set_title("Unshocked Price")
ax.set_xlabel("Index")
ax.set_ylabel("Price")
ax.legend()

plt.show()
