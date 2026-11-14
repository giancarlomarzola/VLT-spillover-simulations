# VLT Spillover Simulations

Simulates price spillover effects from Variable Leverage Token (VLT) rebalancing — modeling how UP/DOWN token rebalance trades move the underlying market price through a synthetic limit orderbook, and how orderbook depth, spread, width, curvature, and resilience shape that impact.

## How it works

- **`core/orderbook.py`** — `Orderbook`: a synthetic bid/ask book defined by depth, spread, width, curvature (`k`), and resilience. Converts a trade size into slippage and tracks depth depletion/replenishment over time.
- **`core/price_spillover_simulations.py`** — `run_simulation`: steps a pair of UP/DOWN leverage tokens through a price series, computing rebalance trades, running them through the orderbook, and feeding the resulting slippage back into the simulated price for the next step.
- **`utils/data_io.py`** — one-off build step: parses the raw Binance trade/rebalance data for each token (keeping only market trades) into a single full-period tick-level parquet, `data/processed/{currency}_tick.parquet`.
- **`utils/data_loader.py`** — `load_data(currency, start, end, freq)`: the only way scripts load market data. Reads the tick parquet once per process, slices the requested window and optionally resamples it (`resample`: last value per right-closed bin, forward-filled).
- **`utils/plotting.py`** — default plot functions: Plotly charts of simulation results (price, leverage, orderbook depth over time) and a matplotlib plot of orderbook shapes (`plot_orderbooks`). **`utils/legacy_plotting.py`** holds the older Plotly helpers used by the dashboard and test bench.

## Setup

```bash
uv sync
```

Requires Python 3.13+.

### Prepare data

Each token's full-period tick data lives in `data/processed/{currency}_tick.parquet`. `load_data` builds it automatically on first use; to (re)build it from the raw data in `data/raw/` explicitly:

```bash
uv run python -m utils.data_io
```

Then load any window at any frequency:

```python
from utils.data_loader import load_data
df = load_data("btc", "2021-05-19 12:00", "2021-05-19 14:00", freq="1s")  # freq=None or "Tick" for raw ticks
```

## Usage

### Interactive dashboard

```bash
uv run streamlit run simulations/dashboard.py
```

Configure currency, frequency, rebalancing strategy, and multiple orderbooks side by side, then compare simulated price, leverage, and depth.

### Scripts

- `simulations/test_bench.py` — run and plot simulations against real market data across several orderbook configurations.
- `simulations/simulation_from_synthetic_price.py` — same, but against a randomly generated synthetic price series.
- `simulations/price_shock_analysis.py` — summary statistics on return distributions for a given currency/frequency.

```bash
uv run python simulations/test_bench.py
```

## Project structure

```
core/          simulation engine (orderbook, price spillover model)
simulations/   entry points: dashboard + standalone scripts
utils/         data loading, paths, plotting helpers
data/          raw/processed data and outputs (figures, results)
```
