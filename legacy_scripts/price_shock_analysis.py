import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.ticker import FuncFormatter, MultipleLocator

from utils.data_loader import load_data
from utils.plotting import TRACE_COLORS

currency = "btc"
frequency = "1s"

start_time = pd.Timestamp("2021-05-19 12:00:00", tz="UTC")
end_time = pd.Timestamp("2021-05-19 14:00:00", tz="UTC")

df = load_data(currency, start_time, end_time, frequency)

df["log_return"] = np.log(df["price"] / df["price"].shift(1))

# Log returns
print("Summary statistics for log returns")
display(df["log_return"].describe().apply("{0:,.5f}".format))
print(f"1st quantile:\t {df['log_return'].quantile(0.01)}")
print(f"99th quantile:\t  {df['log_return'].quantile(0.99)}")


# Get vector with only shock returns (= returns > some cutoff e.g. i-th percentile)
# Maybe for future: Get cutoff value from stable series and apply to crash series

eta_lower = df["log_return"].quantile(0.01)
eta_upper = df["log_return"].quantile(0.99)

df["log_shocks"] = df["log_return"].where((df["log_return"] < eta_lower) | (df["log_return"] > eta_upper), 0)

print("\n\nseries of log_returns that exceed eta:")
display(df[["timestamp", "log_shocks"]][df["log_shocks"] != 0])

# Scatter plot of log returns, highlighting shocks (outside eta_lower/eta_upper) in red
is_shock = df["log_shocks"] != 0

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(
    df["timestamp"][~is_shock],
    df["log_return"][~is_shock],
    color=TRACE_COLORS[0],
    s=1,
    label=f"Normal (n={(~is_shock).sum():,})",
    alpha=0.8,
)
ax.scatter(
    df["timestamp"][is_shock],
    df["log_return"][is_shock],
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

ax.set_xlim(start_time, end_time)
ax.set_ylim(3 * eta_lower, 3 * eta_upper)
ax.set_title("Log Returns with Shock Thresholds")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Log Return")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.legend()

plt.show()


# Load binance data for different time to apply shocks to

start_time = pd.Timestamp("2021-05-19 4:00:00", tz="UTC")
end_time = pd.Timestamp("2021-05-19 6:00:00", tz="UTC")

df_unshocked = load_data(currency, start_time, end_time, frequency)

shock_factor = np.exp(df["log_shocks"].cumsum())
df_unshocked["price_shocked"] = df_unshocked["price"] * shock_factor.to_numpy()


# Plot prices from before crash, during crash, and before crash with artificial shocks
# The series cover different times of day, so align them on minutes since their own start
minutes_elapsed = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 60
minutes_elapsed_unshocked = (df_unshocked["timestamp"] - df_unshocked["timestamp"].iloc[0]).dt.total_seconds() / 60

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    minutes_elapsed_unshocked, df_unshocked["price"],
    color=TRACE_COLORS[0], linewidth=1,
    label="Background Path"
    )
ax.plot(
    minutes_elapsed, df["price"],
    color=TRACE_COLORS[1], linewidth=1,
    label="Shocked Reference Path"
    )
ax.plot(
    minutes_elapsed_unshocked, df_unshocked["price_shocked"],
    color=TRACE_COLORS[2], linewidth=1,
    label="Background Path with injected Shocks"
    )

#ax.set_ylim(0, 1.2 * df["price"].max())
ax.set_title("Application of Shocks from Flash Crash to Background Series")
ax.set_xlim(0, (end_time - start_time).total_seconds() / 60)
ax.xaxis.set_major_locator(MultipleLocator(15))
ax.xaxis.set_major_formatter(FuncFormatter(lambda m, _: f"{int(m) // 60:02d}:{int(m) % 60:02d}"))
ax.set_xlabel("Time Elapsed")
ax.set_ylabel("Price")
ax.legend()

plt.show()
