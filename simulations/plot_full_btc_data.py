# Plot the full 2021-05-19 BTC dataset (not just the 2h analysis window used in test_bench.py).
# Loads the raw Binance trade data the same way data_io.prepare_processed_data does, resamples
# it, caches the result as a processed parquet named after the full time range, and plots it.

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_io import END_TIME, START_TIME, create_currency_df, processed_filename
from utils.paths import DATA_PROCESSED

# Parameters
currency = "btc"
frequency = "500ms"  # resampling frequency for the cached file / plot
reload_raw = False  # True = rebuild the cache from the raw csv.gz files

# Load data (first run reads ~200MB of raw ticks, later runs use the cached parquet)
cached = sorted(DATA_PROCESSED.glob(f"{currency}_{frequency}_processed_full_*.parquet"))

if cached and not reload_raw:
    print(f"Loading cached {cached[0].name}")
    data = pd.read_parquet(cached[0])
else:
    print(f"Loading raw {currency.upper()} tick data for the full day (this takes a while)...")
    ticks = create_currency_df(currency)
    ticks["timestamp"] = pd.to_datetime(ticks["timestamp"], utc=True)
    print(f"Loaded {len(ticks)} ticks from {ticks['timestamp'].min()} to {ticks['timestamp'].max()}")

    # Same resampling as prepare_processed_data: last value per bin, forward-fill gaps
    data = (
        ticks.set_index("timestamp")
        .resample(frequency, closed="right", label="right")
        .last()
        .ffill()
        .reset_index()
    )

    full_name = processed_filename(currency, frequency, data["timestamp"].min(), data["timestamp"].max())
    out_path = DATA_PROCESSED / full_name.replace("_processed_", "_processed_full_")
    data.to_parquet(out_path, index=False)
    print(f"Saved {len(data)} rows to {out_path}")

data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
print(f"{len(data)} rows at {frequency}, {data['timestamp'].min()} to {data['timestamp'].max()}")
print(data[["price", "up_price", "down_price", "leverageUP", "leverageDOWN"]].describe())

# Plot
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=(
        f"{currency.upper()}USDT perpetual price",
        f"{currency.upper()}UP / {currency.upper()}DOWN token prices",
        "Mark-to-market leverage",
    ),
    specs=[[{}], [{"secondary_y": True}], [{}]],
)

x = data["timestamp"]
fig.add_trace(go.Scattergl(x=x, y=data["price"], name="Perpetual", line=dict(color="#636EFA")), row=1, col=1)
fig.add_trace(go.Scattergl(x=x, y=data["up_price"], name="UP token", line=dict(color="#00CC96")), row=2, col=1)
fig.add_trace(
    go.Scattergl(x=x, y=data["down_price"], name="DOWN token", line=dict(color="#EF553B")),
    row=2,
    col=1,
    secondary_y=True,
)
fig.add_trace(go.Scattergl(x=x, y=data["leverageUP"], name="Leverage UP", line=dict(color="#00CC96")), row=3, col=1)
fig.add_trace(
    go.Scattergl(x=x, y=data["leverageDOWN"], name="Leverage DOWN", line=dict(color="#EF553B")), row=3, col=1
)

# Shade the 2h analysis window used by test_bench.py for reference
for row in (1, 2, 3):
    fig.add_vrect(x0=START_TIME, x1=END_TIME, fillcolor="grey", opacity=0.15, line_width=0, row=row, col=1)

fig.update_yaxes(title_text="USDT", row=1, col=1)
fig.update_yaxes(title_text="UP (USDT)", row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text="DOWN (USDT)", row=2, col=1, secondary_y=True)
fig.update_yaxes(title_text="Leverage", row=3, col=1)
fig.update_xaxes(title_text="Time (UTC)", row=3, col=1)
fig.update_layout(
    title=f"{currency.upper()} 2021-05-19, full day at {frequency} (shaded: test bench window)",
    height=900,
    hovermode="x unified",
)
fig.show()
