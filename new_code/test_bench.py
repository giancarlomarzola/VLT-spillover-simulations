# Imports
import pandas as pd

from new_code.plot_utils import plot_results
from new_code.price_spillover_simulations import run_simulation

# Parameters
currency = "btc"
frequency = "15s"

lambda_target = None  # None = boundary rebalancing, float = target rebalancing
lambda_upper = 4  # upper boundary
lambda_lower = 1.25  # lower boundary

# Orderbooks
# width and spread in %
orderbook_formula = "curved"
k = 0.5

orderbooks = {
    "no_orderbook": None,
    "Deep Narrow Tight": {"depth": 50_000_000, "width": 1, "spread": 0.005},
    "Deep Narrow Broad": {"depth": 50_000_000, "width": 1, "spread": 0.05},
    "Deep Wide Tight": {"depth": 50_000_000, "width": 10, "spread": 0.005},
    "Shallow Narrow Tight": {"depth": 5_000_000, "width": 1, "spread": 0.005},
    "Asymmetrical Depth": {
        "depth": (30_000_000, 50_000_000),
        "width": 1,
        "spread": 0.005,
    },
}

# Display options
show_hover_info = True
show_markers = False
plot_resample_freq = "30s"  # None


if __name__ == "__main__":
    # Load pre-processed data
    try:
        print(f"Loading {currency.upper()} at {frequency} frequency...")
        binance_data = pd.read_parquet(
            f"dissertation_data/token_dataframes/{currency}_{frequency}_processed.parquet"
        )
        binance_data["timestamp"] = pd.to_datetime(binance_data["timestamp"], utc=True)
        print(f"Loaded {len(binance_data)} rows")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Pre-processed data not found for {currency} at {frequency} frequency.\n"
            f"Run data_processing.py to prepare the data."
        )

    # Simulation
    results = {}

    for orderbook_name, orderbook in orderbooks.items():
        print(f"Running simulation for {orderbook_name}")
        result = run_simulation(
            lambda_target=lambda_target,
            lambda_upper=lambda_upper,
            lambda_lower=lambda_lower,
            orderbook=orderbook,
            orderbook_formula=orderbook_formula,
            k=k,
            prepared_data=binance_data,
            timestamps=binance_data["timestamp"].values,
        )
        results[orderbook_name] = result

    # Plot all simulations
    print("Plotting the data")
    title_freq = frequency.replace("min", "min ")
    title_prefix = f"{currency.upper()} {title_freq}"

    price_fig, leverage_fig = plot_results(
        results,
        market_price=binance_data["price"].values,
        x_axis=binance_data["timestamp"].values,
        lambda_upper=lambda_upper,
        lambda_lower=lambda_lower,
        show_hover=show_hover_info,
        show_markers=show_markers,
        leverage_timing="After Rebalance",
        title_prefix=title_prefix,
        currency=currency.upper(),
        resample_freq=plot_resample_freq,
    )

    price_fig.show()
    leverage_fig.show()
