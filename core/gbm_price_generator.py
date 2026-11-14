"""
Geometric Brownian motion (GBM) prices.

generate_gbm_price_series: one GBM price path.
GBMPriceGenerator: the input data of a GBM experiment, one path per volatility for each seed.

Every price generator (this one and core.bootstrap_price_generator.BootstrapPriceGenerator) offers
the same interface, which is all core.experiment_runner relies on:
    paths(seed): list of (label, prices) pairs, the price series of one iteration. The label names the
        series in the results (the `price_series` column), e.g. "vol_60" or "btc_bootstrap_60s".
    n_paths: number of paths per iteration.
    config(): the settings that define the paths, saved to the experiment's config.json.
    describe(): one line for the progress printout.
    prepare(): one-off work before the first iteration (e.g. building the pool of returns).
They are plain dataclasses so they can be sent to the worker processes that run iterations in parallel.
To add a new kind of input data, write a new module with a class offering the same five members.
"""

from dataclasses import dataclass

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


@dataclass
class GBMPriceGenerator:
    """
    One GBM path per volatility, all drawn from the same seed so they differ only in volatility.

    volatilities: annualised volatilities, one path each (labelled "vol_<percent>", e.g. "vol_60").
    n_prices: prices per path, including the start price.
    """

    volatilities: list[float]
    n_prices: int
    start_price: float = 100.0
    mu: float = 0.0
    time_step: str = "1s"

    @property
    def n_paths(self) -> int:
        return len(self.volatilities)

    def paths(self, seed: int) -> list[tuple[str, pd.Series]]:
        return [
            (
                f"vol_{round(sigma * 100)}",
                generate_gbm_price_series(
                    n_prices=self.n_prices,
                    start_price=self.start_price,
                    mu=self.mu,
                    sigma=sigma,
                    time_step=self.time_step,
                    seed=seed,
                ),
            )
            for sigma in self.volatilities
        ]

    def config(self) -> dict:
        return {
            "volatilities": self.volatilities,
            "n_prices": self.n_prices,
            "start_price": self.start_price,
            "mu": self.mu,
            "time_step": self.time_step,
        }

    def describe(self) -> str:
        vols = ", ".join(f"{sigma:.0%}" for sigma in self.volatilities)
        return f"{self.n_paths} GBM paths (vol {vols}) of {self.n_prices:,} prices at {self.time_step}"

    def prepare(self) -> None:
        pass  # nothing to build
