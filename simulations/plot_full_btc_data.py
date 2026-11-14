# Plot the full 2021-05-19 BTC dataset (not just the 2h analysis window used in test_bench.py),
# resampled from the full-period tick data.

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import END_TIME, START_TIME, load_data

# Parameters
currency = "btc"
frequency = "500ms"  # resampling frequency for the plot

# Load the full period (start=None, end=None)
data = load_data(currency, start=None, end=None, freq=frequency)
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
