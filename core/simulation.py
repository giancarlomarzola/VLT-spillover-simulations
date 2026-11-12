import numpy as np
import pandas as pd

from core.orderbook import Orderbook
from core.price_process import SECONDS_PER_YEAR
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

    price_array = prices.to_numpy(dtype=float)
    # The loop below is plain Python, where floats and lists are several times faster than
    # numpy scalars and writes to single array cells. Arrays are only built at the end.
    price = price_array.tolist()
    # Seconds since the previous price, used to replenish the orderbook (NaN at t = 0)
    seconds_since_previous_step = prices.index.to_series().diff().dt.total_seconds().tolist()
    for product in vletps:
        product.reset()
    if orderbook is not None:
        orderbook.reset()

    # One row per step: the market columns after the two price columns, and each product's state()
    market_rows = []
    product_rows = [[] for _ in vletps]

    def record(multiplier, slippage, target_total, actual_total):
        if orderbook is not None:
            market_rows.append(
                (multiplier, slippage, target_total, actual_total, orderbook.depth_bid, orderbook.depth_ask)
            )
        else:
            market_rows.append((multiplier, slippage, target_total, actual_total, np.nan, np.nan))
        for product, rows in zip(vletps, product_rows):
            rows.append(tuple(product.state().values()))

    # t = 0: initial state, no trading
    record(1.0, 0.0, 0.0, 0.0)

    m_prev = 1.0  # m_{t-1}
    m_lag = 1.0  # m_{t-2}, needed for the eqn 5 lag

    for t in range(1, len(price)):
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

        record(multiplier, slippage, target_total, actual_total)  # depth after this step's trade, before replenishment

        m_lag = m_prev
        m_prev = multiplier

        # A full book has nothing to replenish, which is the case on most steps
        if orderbook is not None and (
            orderbook.depth_bid != orderbook.depth_bid_full or orderbook.depth_ask != orderbook.depth_ask_full
        ):
            orderbook.replenish(seconds_since_previous_step[t])

    market = np.array(market_rows, dtype=float)
    # state() values, reordered to the PRODUCT_VARS column order
    state_keys = list(vletps[0].state())
    state_order = [state_keys.index(key) for key in PRODUCT_VARS]
    out = np.column_stack(
        [price_array, price_array * market[:, 0], market]
        + [np.array(rows, dtype=float)[:, state_order] for rows in product_rows]
    )
    columns = MARKET_COLUMNS + [f"{prefix}_{name}" for name in names for prefix in PRODUCT_VARS.values()]
    return pd.DataFrame(out, columns=columns, index=prices.index)


def _realised_volatility(price: pd.Series) -> float:
    """Annualised standard deviation of log returns, using the median time step of the index."""
    log_returns = np.log(price).diff().dropna()
    step_seconds = price.index.to_series().diff().dt.total_seconds().median()
    if len(log_returns) < 2 or not step_seconds > 0:
        return np.nan
    return float(log_returns.std() * np.sqrt(SECONDS_PER_YEAR / step_seconds))


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of consecutive True values."""
    padded = np.concatenate(([False], mask, [False])).astype(int)
    edges = np.flatnonzero(np.diff(padded))  # alternating run starts and ends
    return int((edges[1::2] - edges[::2]).max()) if len(edges) else 0


def summarise_run(result: pd.DataFrame, vletps: VLETP | list[VLETP]) -> dict:
    """
    Reduce the output of run_simulation to one row of headline outcomes.

    vletps: the product(s) the simulation was run with, needed for their leverage bands.

    A rebalance is a step with a non-zero executed net trade (actual_total_delta), i.e. the
    trade that reaches the orderbook after netting all products. Sizes are in quote currency.

    Price impact is the slippage caused by a single rebalance (orderbook_effect), as a
    decimal: positive when the net trade is a buy, negative when it is a sell. Averages are
    taken over the rebalances of that sign only, not over every time step. Statistics with
    no rebalances to average over are NaN.

    Price deviation is the cumulative effect of all rebalances on the price so far
    (price_multiplier - 1), as a decimal.

    Realised volatility is the annualised standard deviation of log returns.

    Returns a dict with:
        n_rebalances, longest_rebalance_run (most consecutive steps with a rebalance),
        avg_rebalance_size, total_abs_rebalance_size,
        fill_ratio (executed / wanted rebalance volume; below 1 means the orderbook ran
            out of depth; NaN if no product ever wanted to trade),
        avg_positive_impact, avg_negative_impact, avg_abs_impact,
        max_positive_impact, max_negative_impact (the most negative one),
        final_price_deviation, max_price_deviation, min_price_deviation,
        min_depth_bid, min_depth_ask (lowest depth reached, in quote currency, measured
            right after a trade and before replenishment; NaN without an orderbook),
        and for each product:
            final_nav_<name>,
            wipeout_time_<name> (index value of the step where NAV hit zero, NaT if never),
            n_outside_band_<name> (steps where leverage after the rebalance was still
                outside the leverage band, because the trade was not fully filled),
        realised_vol_raw, realised_vol_simulated,
        realised_vol_ratio (simulated / raw; above 1 means rebalancing added volatility).
    """
    if isinstance(vletps, VLETP):
        vletps = [vletps]

    is_rebalance = result["actual_total_delta"] != 0
    rebalances = result[is_rebalance]
    size = rebalances["actual_total_delta"].abs()
    impact = rebalances["orderbook_effect"]
    positive = impact[impact > 0]
    negative = impact[impact < 0]
    target_volume = result["target_total_delta"].abs().sum()
    price_deviation = result["price_multiplier"] - 1

    summary = {
        "n_rebalances": len(rebalances),
        "longest_rebalance_run": _longest_run(is_rebalance.to_numpy()),
        "avg_rebalance_size": size.mean(),
        "total_abs_rebalance_size": size.sum(),
        "fill_ratio": size.sum() / target_volume if target_volume > 0 else np.nan,
        "avg_positive_impact": positive.mean(),
        "avg_negative_impact": negative.mean(),
        "avg_abs_impact": impact.abs().mean(),
        "max_positive_impact": positive.max(),
        "max_negative_impact": negative.min(),
        "final_price_deviation": price_deviation.iloc[-1],
        "max_price_deviation": price_deviation.max(),
        "min_price_deviation": price_deviation.min(),
        "min_depth_bid": result["depth_bid"].min(),
        "min_depth_ask": result["depth_ask"].min(),
    }
    for product in vletps:
        nav = result[f"nav_{product.name}"]
        leverage_after = result[f"leverage_after_{product.name}"]
        alive = nav > 0
        # Relative tolerance: a rebalance to the boundary lands on it only up to rounding
        outside_band = (leverage_after < product.leverage_lower_bound * (1 - 1e-9)) | (
            leverage_after > product.leverage_upper_bound * (1 + 1e-9)
        )
        summary[f"final_nav_{product.name}"] = nav.iloc[-1]
        summary[f"wipeout_time_{product.name}"] = pd.NaT if alive.all() else alive.idxmin()
        summary[f"n_outside_band_{product.name}"] = int((alive & outside_band).sum())

    vol_raw = _realised_volatility(result["raw_price"])
    vol_simulated = _realised_volatility(result["simulated_price"])
    summary["realised_vol_raw"] = vol_raw
    summary["realised_vol_simulated"] = vol_simulated
    summary["realised_vol_ratio"] = vol_simulated / vol_raw if vol_raw > 0 else np.nan
    return summary


# A run is a feedback loop once rebalancing alone has pushed the price this many times above,
# or below 1/this of, the price without any impact. Normal runs stay within about 2x either
# way; loops run off to the 1e-10 floor or to overflow, so the exact value matters little.
FEEDBACK_LOOP_MULTIPLIER = 10


def flag_feedback_loops(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Flag the runs where a fatal feedback loop took hold: rebalances moved the price, the
    price move forced more rebalances in the same direction, and so on until the price
    multiplier ran away and at least one product was wiped out.

    runs: summarise_run outputs, one row per run (e.g. the saved per-seed files concatenated).

    Returns a DataFrame on the same index with:
        feedback_loop: True for a fatal feedback loop.
        feedback_loop_up / feedback_loop_down: the direction of the loop (price
            multiplier above FEEDBACK_LOOP_MULTIPLIER, or below 1 / FEEDBACK_LOOP_MULTIPLIER).
    """
    max_multiplier = 1 + runs["max_price_deviation"].astype(float)
    min_multiplier = 1 + runs["min_price_deviation"].astype(float)
    wipeout_columns = [column for column in runs.columns if column.startswith("wipeout_time_")]
    any_wipeout = runs[wipeout_columns].notna().any(axis=1)

    up = (max_multiplier >= FEEDBACK_LOOP_MULTIPLIER) & any_wipeout
    down = (min_multiplier <= 1 / FEEDBACK_LOOP_MULTIPLIER) & any_wipeout
    return pd.DataFrame({"feedback_loop": up | down, "feedback_loop_up": up, "feedback_loop_down": down})
