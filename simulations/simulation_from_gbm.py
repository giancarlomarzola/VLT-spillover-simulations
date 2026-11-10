from core.orderbook import Orderbook
from core.price_process import generate_gbm_price_series
from core.simulation import run_simulation
from core.variable_leverage_product import VLETP
from utils.simulation_plots import plot_leverage, plot_orderbook_depth, plot_prices

# Products
gbm_up = VLETP(
    name="GBM_UP",
    start_nav=1_000_000_000,
    start_leverage=2,
    leverage_lower_bound=1.25,
    leverage_upper_bound=4,
    leverage_target=None,
)

# Orderbook
k = 0.3
resilience = 0.2

orderbook = Orderbook(
    name="Deep Narrow Tight",
    depth=50_000_000,
    width=0.01,
    spread=0.005 / 100,
    k=k,
    resilience=resilience,
)

# Price process: raw GBM path, indexed by time elapsed since the start
prices = generate_gbm_price_series(
    n_prices=100_000,
    start_price=50_000,
    mu=0.0,
    sigma=1,
    time_step="5min",
    seed=42,
)

# Simulation: results hold both raw_price (the GBM path) and simulated_price (with VLETP impact)
results = {orderbook.name: run_simulation(gbm_up, orderbook, prices)}

# Plots
plot_resample_freq = None  # e.g. "30s"; None = no resampling

plot_prices(results, resample_freq=plot_resample_freq, title_prefix="GBM").show()
plot_leverage(results, gbm_up, resample_freq=plot_resample_freq, title_prefix="GBM").show()
plot_orderbook_depth(results, resample_freq=plot_resample_freq, title_prefix="GBM").show()
