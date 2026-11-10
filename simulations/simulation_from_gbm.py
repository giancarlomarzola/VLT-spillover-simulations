from core.orderbook import Orderbook
from core.price_process import generate_gbm_price_series
from core.simulation import run_simulation
from core.variable_leverage_product import VLETP
from utils.simulation_plots import plot_leverage, plot_orderbook_depth, plot_prices

# Products
gbm_up = VLETP(
    name="GBM_UP",
    start_nav=10_000_000,
    start_leverage=2,
    leverage_lower_bound=1.25,
    leverage_upper_bound=4,
    leverage_target=None,
)

gbm_down = VLETP(
    name="GBM_DOWN",
    start_nav=10_000_000,
    start_leverage=-2,
    leverage_lower_bound=-1.25,
    leverage_upper_bound=-4,
    leverage_target=None,
)

# Orderbook
baseline_depth = 50_000_000
baseline_width = 0.01
baseline_spread = 0.005 / 100
baseline_k = 0.3
baseline_resilience = 0.2

baseline_orderbook = Orderbook(
    name = "baseline",
    depth = baseline_depth,
    width = baseline_width,
    spread = baseline_spread,
    k = baseline_k,
    resilience = baseline_resilience,
    )

depth_orderbook = Orderbook(
    name = "depth",
    depth = 0.5 * baseline_depth,
    width = baseline_width,
    spread = baseline_spread,
    k = baseline_k,
    resilience = baseline_resilience,
    )

width_orderbook = Orderbook(
    name="width",
    depth = baseline_depth,
    width = 0.5 * baseline_width,
    spread = baseline_spread,
    k = baseline_k,
    resilience = baseline_resilience,
    )

spread_orderbook = Orderbook(
    name="spread",
    depth = baseline_depth,
    width = baseline_width,
    spread = 2 * baseline_spread,
    k = baseline_k,
    resilience = baseline_resilience,
    )

shape_orderbook = Orderbook(
    name="shape",
    depth = baseline_depth,
    width = baseline_width,
    spread = baseline_spread,
    k = 3,
    resilience = baseline_resilience,
    )

resilience_orderbook = Orderbook(
    name="resilience",
    depth = baseline_depth,
    width = baseline_width,
    spread = baseline_spread,
    k = baseline_k,
    resilience = 0.1 * baseline_resilience,
    )

orderbooks = [
    baseline_orderbook, depth_orderbook, width_orderbook, 
    spread_orderbook, shape_orderbook, resilience_orderbook
    ]


# Price process: raw GBM path, indexed by time elapsed since the start
seed = 42

prices_vol_20 = generate_gbm_price_series(
    n_prices=100_000,
    start_price=50_000,
    mu=0.0,
    sigma=0.2,
    time_step="5min",
    seed=seed,
)

prices_vol_40 = generate_gbm_price_series(
    n_prices=100_000,
    start_price=50_000,
    mu=0.0,
    sigma=0.4,
    time_step="5min",
    seed=seed,
)

prices_vol_60 = generate_gbm_price_series(
    n_prices=100_000,
    start_price=50_000,
    mu=0.0,
    sigma=0.6,
    time_step="5min",
    seed=seed,
)

prices_vol_80 = generate_gbm_price_series(
    n_prices=100_000,
    start_price=50_000,
    mu=0.0,
    sigma=0.8,
    time_step="5min",
    seed=seed,
)

prices_vol_100 = generate_gbm_price_series(
    n_prices=100_000,
    start_price=50_000,
    mu=0.0,
    sigma=1,
    time_step="5min",
    seed=seed,
)



# Simulation: results hold both raw_price (the GBM path) and simulated_price (with VLETP impact)
results = {baseline_orderbook.name: run_simulation([gbm_up, gbm_down], baseline_orderbook, prices_vol_100)}

# Plots
plot_resample_freq = None  # e.g. "30s"; None = no resampling

plot_prices(results, resample_freq=plot_resample_freq, title_prefix="GBM").show()
#plot_orderbook_depth(results, resample_freq=plot_resample_freq, title_prefix="GBM").show()
#plot_leverage(results, gbm_up, resample_freq=plot_resample_freq, title_prefix="GBM_UP").show()
#plot_leverage(results, gbm_down, resample_freq=plot_resample_freq, title_prefix="GBM_DOWN").show()
