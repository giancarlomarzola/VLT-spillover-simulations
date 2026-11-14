"""
Prices bootstrapped from historical returns (moving block bootstrap).

block_starts / generate_block_bootstrap_price_series: one resampled price path from a pool of returns.
BootstrapPriceGenerator: the input data of a bootstrap experiment, one resampled path per seed.

Same interface as core.gbm_price_generator.GBMPriceGenerator (see there).
"""

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from core.seeding import iteration_seed


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
        seed: Optional seed for reproducibility (an int or a np.random.SeedSequence)

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


@cache
def _load_returns_pool(returns_path: str, block_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Historical log returns saved at returns_path (a "log_return" column indexed by timestamp), and the
    positions a block of block_length returns can start at without spanning a gap.

    Cached, so each worker process reads the file once rather than once per seed.
    """
    returns = pd.read_parquet(returns_path)["log_return"]
    return returns.to_numpy(dtype=float), block_starts(returns.index, block_length)


@dataclass
class BootstrapPriceGenerator:
    """
    One path per iteration, a moving block bootstrap of historical log returns.

    The returns are the token's trade prices resampled to time_step (last trade per interval,
    forward-filled; see utils.data_loader.log_returns), with the returns inside `outages` dropped.
    Blocks of `block` worth of consecutive returns are kept intact, which preserves the short-range
    dependence of the returns (bid/ask bounce, mean reversion) that an i.i.d. bootstrap would destroy.
    No block spans an outage.

    returns_path: where the pool of returns is saved (built from the tick data by prepare() if missing).
        Kept next to the results it was used for, as it is part of the experiment's setup.
    path_length / block: durations, e.g. "1h" / "60s"; both must be multiples of time_step.
    master_seed: the seed every iteration's random stream is derived from (see core.seeding).
        None = the iteration number is used as the seed directly.
    """

    currency: str
    time_step: str
    path_length: str
    block: str
    start_price: float
    returns_path: Path
    outages: list[tuple[str, str]] = field(default_factory=list)
    master_seed: int | None = None

    @property
    def n_prices(self) -> int:
        return int(pd.Timedelta(self.path_length) / pd.Timedelta(self.time_step)) + 1  # including the start price

    @property
    def block_length(self) -> int:
        return int(pd.Timedelta(self.block) / pd.Timedelta(self.time_step))  # returns per block

    @property
    def label(self) -> str:
        return f"{self.currency}_bootstrap_{self.block}"

    @property
    def n_paths(self) -> int:
        return 1

    def paths(self, seed: int) -> list[tuple[str, pd.Series]]:
        returns, starts = _load_returns_pool(str(self.returns_path), self.block_length)
        prices = generate_block_bootstrap_price_series(
            returns,
            starts,
            n_prices=self.n_prices,
            block_length=self.block_length,
            start_price=self.start_price,
            time_step=self.time_step,
            seed=iteration_seed(self.master_seed, seed),
        )
        return [(self.label, prices)]

    def config(self) -> dict:
        config = {
            "type": "moving block bootstrap",
            "currency": self.currency,
            "time_step": self.time_step,
            "outages": self.outages,
            "start_price": self.start_price,
            "path_length": self.path_length,
            "n_prices": self.n_prices,
            "block": self.block,
            "block_length": self.block_length,
        }
        if self.master_seed is not None:  # left out otherwise, so older experiments' config.json still match
            config["master_seed"] = self.master_seed
        return config

    def describe(self) -> str:
        return (
            f"one {self.path_length} {self.currency.upper()} bootstrapped path "
            f"({self.n_prices:,} prices at {self.time_step}, blocks of {self.block})"
        )

    def prepare(self) -> None:
        """Build the pool of returns from the tick data and save it to returns_path, unless it already exists."""
        returns_path = Path(self.returns_path)
        if returns_path.exists():
            return
        from utils.data_loader import log_returns  # only the main process needs the tick data

        returns = log_returns(self.currency, self.time_step, exclude=self.outages)
        returns_path.parent.mkdir(parents=True, exist_ok=True)
        returns.to_frame().to_parquet(returns_path)
        print(
            f"{len(returns):,} {self.time_step} returns saved to {returns_path} "
            f"({returns.index[0]} to {returns.index[-1]}, outages excluded: {self.outages})"
        )
