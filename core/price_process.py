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
