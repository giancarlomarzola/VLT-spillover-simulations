# Imports
from core.orderbook import Orderbook
from core.simulation import run_simulation
from utils.data_loader import END_TIME, START_TIME, load_data, products_from_data
from utils.plotting import plot_leverage, plot_orderbook_depth, plot_prices

# Parameters
currency = "btc"
frequency = "1s"  # resampling frequency, "Tick" = raw ticks
start_time = START_TIME
end_time = END_TIME

lambda_target = None  # None = boundary rebalancing, float = target rebalancing
lambda_upper = 4  # upper boundary
lambda_lower = 1.25  # lower boundary

k = 0.3
resilience = "5s"  # half-life of orderbook depth recovery

deep, shallow = 50_000_000, 10_000_000
narrow, wide = 1 / 100, 10 / 100
tight, broad = 0.5 / 10_000, 5 / 10_000

# Display options
show_hover_info = True
plot_resample_freq = None
leverage_after_rebalance = False  # False = leverage before each step's rebalance


# Orderbooks
# width and spread in decimal form
orderbooks = [
    None,
    Orderbook(
        name="Deep Narrow Tight",
        depth=deep,
        width=narrow,
        spread=tight,
        k=k,
        resilience=resilience,
    ),
    Orderbook(
        name="Deep Narrow Broad",
        depth=deep,
        width=narrow,
        spread=broad,
        k=k,
        resilience=resilience,
    ),
    Orderbook(
        name="Deep Wide Tight",
        depth=deep,
        width=wide,
        spread=tight,
        k=k,
        resilience=resilience,
    ),
    Orderbook(
        name="Shallow Narrow Tight",
        depth=shallow,
        width=narrow,
        spread=tight,
        k=k,
        resilience=resilience,
    ),
    Orderbook(
        name="Asymmetrical Depth",
        depth=(30_000_000, deep),
        width=narrow,
        spread=tight,
        k=k,
        resilience=resilience,
    ),
]


# Load data
print(f"Loading {currency.upper()} at {frequency} frequency...")
binance_data = load_data(currency, start_time, end_time, frequency)
print(f"Loaded {len(binance_data)} rows")

# Products (UP and DOWN, seeded from the issuer holdings at the start of the window)
products = products_from_data(binance_data, currency, lambda_lower, lambda_upper, lambda_target)
prices = binance_data.set_index("timestamp")["price"]

# Simulation
results = {}

for orderbook in orderbooks:
    orderbook_name = orderbook.name if orderbook is not None else "No Orderbook"
    print(f"Running simulation for {orderbook_name}")
    result = run_simulation(products, orderbook, prices)
    results[orderbook_name] = result

# Plot all simulations
print("Plotting the data")
title_freq = frequency.replace("min", "min ")
title_prefix = f"{currency.upper()} {title_freq}"

price_fig = plot_prices(
    results,
    resample_freq=plot_resample_freq,
    show_hover=show_hover_info,
    title_prefix=title_prefix,
)
leverage_fig = plot_leverage(
    results,
    products,
    after_rebalance=leverage_after_rebalance,
    resample_freq=plot_resample_freq,
    show_hover=show_hover_info,
    title_prefix=title_prefix,
)

price_fig.show()
leverage_fig.show()

# Plot orderbook depth over time (if orderbooks were used)
print("\nPlotting orderbook depth")
if any(name != "No Orderbook" for name in results):
    depth_fig = plot_orderbook_depth(
        results,
        resample_freq=plot_resample_freq,
        show_hover=show_hover_info,
        title_prefix=title_prefix,
    )
    depth_fig.show()
