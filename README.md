# VLT Spillover Simulations

Simulates price spillover effects from Variable Leverage Token (VLT) rebalancing — modeling how UP/DOWN token rebalance trades move the underlying market price through a synthetic limit orderbook, and how orderbook depth, spread, width, curvature, and resilience shape that impact.

## How it works

- **`core/orderbook.py`** — `Orderbook`: a synthetic bid/ask book defined by depth, spread, width, curvature (`k`), and resilience. Converts a trade size into slippage and tracks depth depletion/replenishment over time.
- **`core/price_spillover_simulations.py`** — `run_simulation`: steps a pair of UP/DOWN leverage tokens through a price series, computing rebalance trades, running them through the orderbook, and feeding the resulting slippage back into the simulated price for the next step.
- **`utils/data_io.py`** — loads raw Binance trade/rebalance data and resamples it into the frequencies used by the simulations.
- **`utils/plotting.py`**, **`utils/orderbook_plots.py`** — shared Plotly chart helpers for price, leverage, and orderbook depth.

## Setup

```bash
uv sync
```

Requires Python 3.13+.

### Prepare data

Processed parquet files are expected under `data/processed/`. To regenerate them from raw data in `data/raw/`:

```bash
uv run python utils/data_io.py
```

## Usage

### Interactive dashboard

```bash
uv run streamlit run simulations/dashboard.py
```

Configure currency, frequency, rebalancing strategy, and multiple orderbooks side by side, then compare simulated price, leverage, and depth.

### Scripts

- `simulations/test_bench.py` — run and plot simulations against real processed market data across several orderbook configurations.
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
