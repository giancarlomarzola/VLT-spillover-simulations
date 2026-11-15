# VLT Spillover Simulations

Simulates the price spillover of Variable Leverage Token (VLT) rebalancing: how the rebalance trades of UP/DOWN leveraged products move the price of the underlying through a synthetic limit orderbook, and how the orderbook's depth, spread, width, curvature and resilience shape that impact.

Simulations run either on historical market data (Binance, 2021-05-19) or on synthetic price paths (GBM, or block-bootstrapped historical returns), from single runs explored in a dashboard to experiments of thousands of seeded runs summarised across seeds.

All simulations assume markets trading 24/7 and rebalances executed immediately and in full.

## The model

- **`core/variable_leverage_product.py`**: `VLETP`, one variable leverage product (e.g. a BTC UP or DOWN token). Its leverage drifts freely inside a band `[leverage_lower_bound, leverage_upper_bound]`; when it leaves the band, the product rebalances to the nearest boundary or to a target leverage. The sign of the band sets the direction (long UP / short DOWN).
- **`core/orderbook.py`**: `Orderbook`, a synthetic bid/ask book defined by depth, width (slippage at full depth), spread, curvature `k` and resilience (half-life of depth recovery). Turns a trade size into slippage and tracks how trades deplete depth and how it recovers over time. Every parameter can differ between the bid and ask side.
- **`core/simulation.py`**:
  - `run_simulation(vletps, orderbook, prices)` steps one or more products along a price path. Each step their rebalance trades are netted, executed against the orderbook, and the resulting slippage moves the price used at the next step, which can trigger further rebalances. `orderbook=None` is the reference run without price impact.
  - `summarise_run` reduces a run to one row of headline outcomes (rebalances, price impact, final price deviation, wipeouts, realised volatility, ...).
  - `flag_feedback_loops` flags runs where rebalancing and price moves fed each other until a product was wiped out.

## Price inputs

- **Historical data**:
  - `utils/data_io.py` is a one-off build step. It parses the raw Binance trade and leverage-token rebalance files in `data/raw/` into one tick-level parquet per token, `data/processed/{currency}_tick.parquet` (currently btc, eth, sushi).
  - `utils/data_loader.py` is how scripts read that data:
    - `load_data(currency, start, end, freq)` returns a window, optionally resampled.
    - `products_from_data` builds the UP/DOWN products from the token data.
    - `log_returns` gives log returns on a regular grid, with outage windows cut out.
- **`core/gbm_price_generator.py`**:
  - `generate_gbm_price_series` draws one GBM path.
  - `GBMPriceGenerator` is the input of a GBM experiment: one path per volatility for each seed.
- **`core/bootstrap_price_generator.py`**:
  - `generate_block_bootstrap_price_series` resamples historical returns in blocks (a moving block bootstrap). Keeping blocks intact preserves short-range dependence such as bid/ask bounce. No block spans a data outage.
  - `BootstrapPriceGenerator` is the input of a bootstrap experiment: one resampled path per seed.
- `utils/coindesk_data.py` fetches historical futures trades from the CoinDesk API into `data/raw/coindesk_api/`. It needs an API key in `utils/api_key.py`, which is gitignored.

## Experiments

An experiment runs many iterations of one setup. Each iteration is one seed: the seed draws the price paths, and every path is run against every orderbook.

- **`core/experiment_runner.py`**:
  - `run_experiment` runs the seeds not yet saved, in parallel worker processes, then builds the summary tables. `n_iterations` is a target total, so running again completes an interrupted experiment or adds seeds after raising it.
  - `simulate_seed` runs one iteration.
  - `load_setup(experiment_dir)` rebuilds an experiment's products, orderbooks and price generator from its `config.json`, so any saved seed can be re-run in full.
- **`core/summary_tables.py`**: `build_summary_tables` averages every (price series, orderbook) cell across seeds, with standard deviations, into `summary_table.xlsx`. Runs that fell into a feedback loop are counted but left out of the averages.
- **`simulations/products_and_orderbooks.py`**: shared building blocks for experiments:
  - `vletp_pair(asset)` gives the UP/DOWN pair.
  - `orderbook_sensitivities()` gives no orderbook, a baseline book, and the baseline with one parameter changed at a time.
- **One script per experiment**, holding its settings:
  - `simulations/run_gbm_experiment.py`: GBM paths at several volatilities.
  - `simulations/run_btc_bootstrap_experiment.py`: 1h paths bootstrapped from the 2021-05-19 BTC returns at 100ms.

  To set up a new experiment, copy one of them and give it a new `experiment` name.

Each experiment is saved in its own folder, `data/outputs/GBM_simulations/<experiment>/` or `data/outputs/BTC_bootstrap_simulations/<experiment>/`:

```
config.json          the setup the results belong to; a different setup is refused, so results are never mixed
iterations/          one parquet file of run summaries per seed (seed_000042.parquet, ...)
summary_table.xlsx   averages ("summary") and standard deviations ("stdev") across seeds
returns.parquet      bootstrap experiments only: the pool of returns the paths are drawn from
```

To inspect one seed in full (every time step of every run):

```python
from core.experiment_runner import load_setup, simulate_seed
from utils.paths import GBM_SIMULATIONS
from utils.plotting import plot_leverage, plot_prices

setup = load_setup(GBM_SIMULATIONS / "1s_2kappas")
summary, results = simulate_seed(42, **setup, keep_results=True)
plot_prices(results["vol_60", "baseline"]).show()
plot_leverage(results["vol_60", "baseline"], setup["vletps"]).show()
```

## Setup

```bash
uv sync
```

Requires Python 3.13+.

The tick parquet for a token is built automatically the first time it is loaded. To (re)build it from `data/raw/` explicitly:

```bash
uv run python -m utils.data_io
```

```python
from utils.data_loader import load_data
df = load_data("btc", "2021-05-19 12:00", "2021-05-19 14:00", freq="1s")  # freq=None or "Tick" for raw ticks
```

## Usage

Run everything from the repo root.

### Experiments

```bash
uv run python -m simulations.run_gbm_experiment
uv run python -m simulations.run_btc_bootstrap_experiment
```

### Interactive dashboard

```bash
uv run streamlit run legacy_scripts/dashboard.py
```

Pick a currency, frequency and rebalancing rule (leverage band, boundary or target rebalancing), set up several orderbooks side by side, and compare simulated price, leverage and orderbook depth on the default historical window (2021-05-19 11:45 to 13:45 UTC).

### Other scripts

- `legacy_scripts/test_bench.py`: run and plot simulations on historical data against several orderbook configurations (the script version of the dashboard).
- `simulations/orderbook_archetypes.py`: plot the slippage curves of a set of orderbook archetypes (deep/shallow, narrow/wide, tight/broad).
- `simulations/plot_full_btc_data.py`: plot the full 2021-05-19 BTC data (price, UP/DOWN token prices, leverage) for visual inspection.
- `legacy_scripts/price_shock_analysis.py`: statistics and plots of the return distribution for a currency and frequency.
- `simulations/quant_minds_plots_btc.py`: figures and tables for the QuantMinds presentation, from the BTC bootstrap experiment `100ms_4h_block1s`.
- `simulations/quant_minds_plots_gbm.py`: GBM single-path price figure for the QuantMinds presentation, from the GBM experiment `1s_2kappas`.

### Plotting

`utils/plotting.py` holds the plot functions:
- Interactive Plotly charts of simulation results: `plot_prices`, `plot_leverage`, `plot_orderbook_depth`.
- A matplotlib plot of orderbook shapes: `plot_orderbooks`.

### Tests

```bash
uv run pytest
```

## Project structure

```
core/           the model (products, orderbook, simulation), price generators and the experiment runner
simulations/    entry points: experiment scripts and standalone analysis scripts
legacy_scripts/ older scripts on historical data: dashboard, test bench, price shock analysis
utils/          data building and loading, paths, plotting
tests/          pytest suite for core/
data/           raw/ and processed/ market data; outputs/ for experiments and figures (not in git)
```
