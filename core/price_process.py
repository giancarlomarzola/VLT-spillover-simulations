import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365 * 24 * 60 * 60  # markets trade around the clock


def generate_gbm_price_series(n_prices, start_price=100.0, mu=0.0, sigma=0.2, time_step="1s", seed=None):
    """
    Generate a price series following geometric Brownian motion:

        S_{t+dt} = S_t * exp((mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z),  Z ~ N(0, 1)

    Args:
        n_prices: Number of prices to simulate (including the start price)
        start_price: Initial price S_0
        mu: Annualised drift
        sigma: Annualised volatility
        time_step: Time between consecutive prices, as a pandas frequency string or
            Timedelta (e.g. "1s", "500ms", "1min"). Converted to years (365-day year) for dt.
        seed: Optional seed for reproducibility

    Returns:
        pd.Series: n_prices simulated prices, indexed by time elapsed since the start of
            the simulation (a TimedeltaIndex: 0, time_step, 2 * time_step, ...)
    """
    if n_prices < 1:
        raise ValueError("n_prices must be at least 1")
    if start_price <= 0:
        raise ValueError("start_price must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    time_step = pd.Timedelta(time_step)
    if time_step <= pd.Timedelta(0):
        raise ValueError("time_step must be positive")

    dt = time_step.total_seconds() / SECONDS_PER_YEAR

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n_prices - 1)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_prices = np.log(start_price) + np.concatenate(([0.0], np.cumsum(log_returns)))

    elapsed = pd.timedelta_range(start=0, periods=n_prices, freq=time_step, name="elapsed")
    return pd.Series(np.exp(log_prices), index=elapsed, name="price")


def block_starts(timestamps, block_length):
    """
    Positions where a block of `block_length` consecutive returns can start without spanning a gap.

    Args:
        timestamps: DatetimeIndex (or array) of the returns' timestamps, on a regular grid with windows cut out
            (e.g. the index of utils.data_loader.log_returns).
        block_length: number of returns per block.

    Returns:
        np.ndarray of positions i such that returns i .. i + block_length - 1 are contiguous on the grid.
    """
    if block_length < 1:
        raise ValueError("block_length must be at least 1")
    ns = np.asarray(pd.DatetimeIndex(timestamps).asi8)
    if len(ns) < block_length:
        return np.array([], dtype=np.int64)
    step = np.median(np.diff(ns)) if len(ns) > 1 else 0
    span = ns[block_length - 1 :] - ns[: len(ns) - block_length + 1]
    return np.flatnonzero(span == (block_length - 1) * step)


def generate_block_bootstrap_price_series(
    log_returns, starts, n_prices, block_length, start_price=100.0, time_step="100ms", seed=None
):
    """
    Generate a price series by resampling historical log returns in blocks (moving block bootstrap):

        S_t = S_0 * exp(cumulative sum of the resampled returns)

    Blocks of `block_length` consecutive returns are drawn with replacement and placed end to end, and the
    last block is cut short to give exactly n_prices - 1 returns. Keeping each block intact preserves the
    short-range dependence of the returns within it (e.g. bid/ask bounce), which an i.i.d. bootstrap would
    destroy.

    Args:
        log_returns: historical log returns (array-like), on a grid with spacing `time_step`.
        starts: positions a block may start at (see block_starts), so no block spans a gap in the data.
        n_prices: Number of prices to simulate (including the start price)
        block_length: Number of consecutive returns per block
        start_price: Initial price S_0
        time_step: Time between consecutive prices, as a pandas frequency string or Timedelta. Should match
            the spacing of log_returns; it only sets the index of the output.
        seed: Optional seed for reproducibility

    Returns:
        pd.Series: n_prices prices, indexed by time elapsed since the start of the simulation
            (a TimedeltaIndex: 0, time_step, 2 * time_step, ...)
    """
    if n_prices < 1:
        raise ValueError("n_prices must be at least 1")
    if start_price <= 0:
        raise ValueError("start_price must be positive")
    if block_length < 1:
        raise ValueError("block_length must be at least 1")
    starts = np.asarray(starts)
    if n_prices > 1 and len(starts) == 0:
        raise ValueError("no valid block starts: the data has no run of block_length contiguous returns")
    time_step = pd.Timedelta(time_step)
    if time_step <= pd.Timedelta(0):
        raise ValueError("time_step must be positive")

    log_returns = np.asarray(log_returns, dtype=float)
    n_returns = n_prices - 1
    n_blocks = -(-n_returns // block_length)  # ceiling division

    rng = np.random.default_rng(seed)
    chosen = rng.choice(starts, size=n_blocks, replace=True) if n_blocks else np.array([], dtype=np.int64)
    positions = (chosen[:, None] + np.arange(block_length)).ravel()[:n_returns]
    sampled = log_returns[positions]
    log_prices = np.log(start_price) + np.concatenate(([0.0], np.cumsum(sampled)))

    elapsed = pd.timedelta_range(start=0, periods=n_prices, freq=time_step, name="elapsed")
    return pd.Series(np.exp(log_prices), index=elapsed, name="price")
