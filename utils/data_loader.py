"""Load market data for a token: a time window of the full-period tick data, optionally resampled.

Raw data is parsed once per token by `utils.data_io.build_tick_data` into `data/processed/{currency}_tick.parquet`.
This module reads that file at most once per process (cached in memory), then slices and resamples on demand:

    from utils.data_loader import load_data
    df = load_data("btc", "2021-05-19 12:00", "2021-05-19 14:00", freq="1s")
"""

from functools import cache

import pandas as pd

from core.variable_leverage_product import VLETP
from utils.data_io import build_tick_data, tick_path

# Frequencies offered by the dashboard ("Tick" = no resampling)
FREQUENCIES = ["Tick", "50ms", "500ms", "1s", "15s", "30s", "1min"]

# Default analysis window (UTC)
START_TIME = pd.Timestamp("2021-05-19 11:45:00", tz="UTC")
END_TIME = pd.Timestamp("2021-05-19 13:45:00", tz="UTC")


def _to_utc(ts):
    """Parse a timestamp, treating naive values as UTC."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


@cache
def _load_ticks(currency):
    """Read the full-period tick data for a token, building it from raw files if it doesn't exist yet."""
    path = tick_path(currency)
    if not path.exists():
        print(f"{path.name} not found, building it from raw data (one-off)...")
        build_tick_data(currency)

    data = pd.read_parquet(path)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data


def load_ticks(currency):
    """Full-period tick data for a token (a copy, safe to modify)."""
    return _load_ticks(currency).copy()


def resample(data, freq, start=None, end=None):
    """Resample tick data onto a regular grid: last value per bin, forward-filled across empty bins.

    Bins are closed and labelled on the right, so the row at time t holds the last tick in (t - freq, t].
    If start/end are given, the output is restricted to bins labelled in [start, end]. Ticks before `start`
    are used to seed the forward fill, so pass the full tick history rather than a pre-sliced window.

    Args:
        data: tick-level DataFrame with a UTC "timestamp" column, sorted by time.
        freq: fixed-width pandas frequency string (e.g. "500ms", "1s", "1min"). None or "Tick" returns the ticks
            unchanged.
        start, end: optional window bounds (UTC, naive values are treated as UTC).
    """
    start = None if start is None else _to_utc(start)
    end = None if end is None else _to_utc(end)

    if freq is None or freq == "Tick":
        return _slice(data, start, end, seed=False)

    # Keep the last tick before the window as the forward-fill seed; earlier ticks can't affect the result
    window = _slice(data, start, end, seed=True)
    resampled = window.set_index("timestamp").resample(freq, closed="right", label="right").last()
    if end is not None and len(resampled) and resampled.index[-1] < end:
        # No ticks near the end of the window (e.g. an exchange outage): extend the grid so it still reaches `end`
        grid = pd.date_range(resampled.index[0], end, freq=freq)
        resampled = resampled.reindex(resampled.index.union(grid))
    resampled = resampled.ffill().rename_axis("timestamp").reset_index()

    if start is not None:
        resampled = resampled[resampled["timestamp"] >= start]
    if end is not None:
        resampled = resampled[resampled["timestamp"] <= end]
    return resampled.reset_index(drop=True)


def _slice(data, start, end, seed):
    """Rows with start <= timestamp <= end; with seed=True also the last row before start."""
    timestamps = data["timestamp"]
    lo = 0 if start is None else int(timestamps.searchsorted(start, side="left"))
    hi = len(data) if end is None else int(timestamps.searchsorted(end, side="right"))
    if seed:
        lo = max(lo - 1, 0)
    return data.iloc[lo:hi].reset_index(drop=True)


def load_data(currency, start=START_TIME, end=END_TIME, freq=None):
    """Load a window of market data for a token, optionally resampled.

    Args:
        currency: token ticker, e.g. "btc", "eth", "sushi".
        start, end: window bounds (UTC, inclusive). None = from the first / to the last available tick.
            Defaults to the analysis window START_TIME..END_TIME.
        freq: resampling frequency (e.g. "1s"). None or "Tick" = raw ticks.

    Returns:
        DataFrame with a UTC "timestamp" column plus price, latency and rebalance/leverage columns.
    """
    ticks = _load_ticks(currency)

    first, last = ticks["timestamp"].iloc[0], ticks["timestamp"].iloc[-1]
    start_utc = first if start is None else _to_utc(start)
    end_utc = last if end is None else _to_utc(end)
    if start_utc >= end_utc:
        raise ValueError(f"start ({start_utc}) must be before end ({end_utc})")
    if start_utc < first or end_utc > last:
        raise ValueError(
            f"Requested window [{start_utc}, {end_utc}] falls outside the available data [{first}, {last}] "
            f"for {currency}"
        )

    return resample(ticks, freq, start_utc, end_utc)


def products_from_data(data, currency, leverage_lower_bound, leverage_upper_bound, leverage_target=None):
    """Build the token's UP and DOWN products from the issuer holdings in the first row of `data`.

    Args:
        data: DataFrame from load_data (needs price, nTokensUP/DOWN, up_price/down_price, BasketUP/DOWN).
        currency: token ticker, used to name the products (e.g. "btc" -> BTCUP, BTCDOWN).
        leverage_lower_bound, leverage_upper_bound: leverage band, as positive magnitudes. The DOWN
            product gets the same band mirrored below zero.
        leverage_target: None = nearest-boundary rebalancing, a positive number = rebalance to that
            leverage (mirrored for DOWN).

    Returns:
        [UP product, DOWN product]
    """
    first = data.iloc[0]
    products = []
    for side, sign in (("UP", 1), ("DOWN", -1)):
        products.append(
            VLETP.from_basket(
                name=f"{currency.upper()}{side}",
                n_tokens=first[f"nTokens{side}"],
                token_price=first[f"{side.lower()}_price"],
                basket=first[f"Basket{side}"],
                underlying_price=first["price"],
                leverage_lower_bound=sign * leverage_lower_bound,
                leverage_upper_bound=sign * leverage_upper_bound,
                leverage_target=None if leverage_target is None else sign * leverage_target,
            )
        )
    return products
