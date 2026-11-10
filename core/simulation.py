import numpy as np
import pandas as pd

from core.orderbook import Orderbook
from core.variable_leverage_product import VLETP

# VLETP.state() key -> prefix of the per-product output column, e.g. "nav_BTCUP"
PRODUCT_VARS = {
    "v": "nav",
    "x": "exposure",  # notional before rebalance
    "x_star": "exposure_after",  # notional after rebalance
    "lam": "leverage",  # leverage before rebalance
    "lam_star": "leverage_after",  # leverage after rebalance
    "target_delta": "target_delta",  # rebalance the product wanted
    "actual_delta": "actual_delta",  # rebalance actually executed
}

MARKET_COLUMNS = [
    "raw_price",
    "simulated_price",
    "price_multiplier",
    "orderbook_effect",
    "target_total_delta",
    "actual_total_delta",
    "depth_bid",
    "depth_ask",
]


def run_simulation(
    vletps: VLETP | list[VLETP],
    orderbook: Orderbook | None,
    prices: pd.Series,
) -> pd.DataFrame:
    """
    Simulate one or more VLETPs rebalancing against a shared orderbook along a price path.

    vletps: a single VLETP or a list of them. Their rebalance trades are netted each step
        before hitting the orderbook. Names must be unique.
    orderbook: Orderbook the net rebalance trade executes against, or None for no price impact.
    prices: price path without any VLETP impact, as a Series indexed by time: either a
        DatetimeIndex (historical data) or a TimedeltaIndex of time elapsed since the start
        (e.g. a simulated GBM path).

    The products and the orderbook are reset at the start, so the same objects can be reused
    across runs. They are left in their end-of-simulation state.

    Slippage and the price multiplier follow the one-period execution lag of eqn 5: a
    rebalance at t only moves the price used at t+1.

    Returns a DataFrame on the same index as `prices` with:
        raw_price: the input price path, untouched.
        simulated_price: raw_price * price_multiplier, i.e. the price including the
            cumulative impact of the products' rebalancing.
        price_multiplier, orderbook_effect (slippage of this step's net trade),
        target_total_delta, actual_total_delta, depth_bid, depth_ask.
        <variable>_<product name> for each product and each variable in PRODUCT_VARS,
            e.g. nav_BTCUP, leverage_after_BTCUP.
    """
    if isinstance(vletps, VLETP):
        vletps = [vletps]
    vletps = list(vletps)
    if not vletps:
        raise ValueError("vletps must contain at least one product")
    names = [product.name for product in vletps]
    if len(set(names)) != len(names):
        raise ValueError(f"VLETP names must be unique, got {names}")

    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pandas Series indexed by time, got {type(prices)}")
    if not isinstance(prices.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        raise TypeError(f"prices must have a DatetimeIndex or TimedeltaIndex, got {type(prices.index)}")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices index must be sorted in increasing time order")
    if len(prices) == 0:
        raise ValueError("prices must not be empty")

    price = prices.to_numpy(dtype=float)
    # Seconds since the previous price, used to replenish the orderbook (NaN at t = 0)
    step_seconds = prices.index.to_series().diff().dt.total_seconds().to_numpy()

    for product in vletps:
        product.reset()
    if orderbook is not None:
        orderbook.reset()

    columns = MARKET_COLUMNS + [f"{prefix}_{name}" for name in names for prefix in PRODUCT_VARS.values()]
    col = {name: i for i, name in enumerate(columns)}
    n = len(price)
    out = np.zeros((n, len(columns)))

    def record(t):
        for product in vletps:
            for key, value in product.state().items():
                out[t, col[f"{PRODUCT_VARS[key]}_{product.name}"]] = value
        out[t, col["depth_bid"]] = orderbook.depth_bid if orderbook is not None else np.nan
        out[t, col["depth_ask"]] = orderbook.depth_ask if orderbook is not None else np.nan

    # t = 0: initial state, no trading
    out[0, col["raw_price"]] = price[0]
    out[0, col["simulated_price"]] = price[0]
    out[0, col["price_multiplier"]] = 1.0
    record(0)

    m_lag = 1.0  # m_{t-2}, needed for the eqn 5 lag

    for t in range(1, n):
        m_prev = out[t - 1, col["price_multiplier"]]  # m_{t-1}

        # Return of the impacted price. Guard against zero/corrupted prices or multipliers
        if price[t - 1] == 0 or price[t] == 0 or m_lag == 0 or m_prev == 0:
            ret = 0.0
        else:
            ret = (price[t] * m_prev) / (price[t - 1] * m_lag) - 1

        # Mark every product to market and net their desired rebalance trades
        target_total = 0.0
        for product in vletps:
            product.apply_return(ret)
            target_total += product.calculate_rebalance()

        # Execute the net trade; if it exceeds available depth, every product is scaled back (eqn 14)
        effective_price = price[t] * m_prev
        if orderbook is not None and target_total != 0 and effective_price != 0:
            slippage, scale = orderbook.execute_transaction(target_total / effective_price, effective_price)
        else:
            slippage, scale = 0.0, 1.0

        actual_total = 0.0
        for product in vletps:
            actual_total += product.execute_rebalance(scale)

        # Clamp to keep the multiplier positive (a non-positive one would give NaN/inf next step)
        multiplier = max((1 + slippage) * m_prev, 1e-10)

        out[t, col["raw_price"]] = price[t]
        out[t, col["simulated_price"]] = price[t] * multiplier
        out[t, col["price_multiplier"]] = multiplier
        out[t, col["orderbook_effect"]] = slippage
        out[t, col["target_total_delta"]] = target_total
        out[t, col["actual_total_delta"]] = actual_total
        record(t)  # depth recorded after this step's trade, before replenishment

        m_lag = m_prev

        if orderbook is not None:
            orderbook.replenish(step_seconds[t])

    return pd.DataFrame(out, columns=columns, index=prices.index)
