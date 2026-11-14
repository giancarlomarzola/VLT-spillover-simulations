import numpy as np
import pandas as pd
import pytest

from core.price_process import (
    SECONDS_PER_YEAR,
    block_starts,
    generate_block_bootstrap_price_series,
    generate_gbm_price_series,
)


def test_shape_and_index():
    prices = generate_gbm_price_series(5, start_price=250, time_step="500ms", seed=0)
    assert isinstance(prices, pd.Series)
    assert len(prices) == 5
    assert prices.iloc[0] == pytest.approx(250)
    assert isinstance(prices.index, pd.TimedeltaIndex)
    assert prices.index[0] == pd.Timedelta(0)
    assert (prices.index.to_series().diff().dropna() == pd.Timedelta("500ms")).all()
    assert (prices > 0).all()


def test_single_price_is_the_start_price():
    prices = generate_gbm_price_series(1, start_price=42)
    assert prices.tolist() == pytest.approx([42])


def test_seed_makes_the_path_reproducible():
    first = generate_gbm_price_series(100, seed=7)
    second = generate_gbm_price_series(100, seed=7)
    other = generate_gbm_price_series(100, seed=8)
    pd.testing.assert_series_equal(first, second)
    assert not np.allclose(first, other)


def test_zero_volatility_gives_pure_drift():
    prices = generate_gbm_price_series(4, start_price=100, mu=0.1, sigma=0, time_step="1D")
    years = np.arange(4) * 24 * 60 * 60 / SECONDS_PER_YEAR
    assert prices.to_numpy() == pytest.approx(100 * np.exp(0.1 * years))


def test_zero_drift_and_volatility_gives_flat_path():
    prices = generate_gbm_price_series(10, start_price=100, mu=0, sigma=0)
    assert prices.to_numpy() == pytest.approx(np.full(10, 100.0))


def test_realised_volatility_matches_sigma():
    prices = generate_gbm_price_series(200_000, sigma=0.5, time_step="1min", seed=1)
    log_returns = np.log(prices).diff().dropna()
    realised = log_returns.std() * np.sqrt(SECONDS_PER_YEAR / 60)
    assert realised == pytest.approx(0.5, rel=0.02)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_prices": 0},
        {"start_price": 0},
        {"start_price": -1},
        {"sigma": -0.1},
        {"time_step": "0s"},
        {"time_step": "-1s"},
    ],
)
def test_invalid_arguments_rejected(kwargs):
    arguments = {"n_prices": 10, **kwargs}
    with pytest.raises(ValueError):
        generate_gbm_price_series(**arguments)


def _grid_with_gap():
    """Timestamps 0..4s and 10..12s on a 1s grid: a gap between 4s and 10s."""
    seconds = [0, 1, 2, 3, 4, 10, 11, 12]
    return pd.DatetimeIndex(pd.Timestamp("2021-05-19", tz="UTC") + pd.to_timedelta(seconds, unit="s"))


def test_block_starts_skip_blocks_spanning_a_gap():
    timestamps = _grid_with_gap()
    assert block_starts(timestamps, 1).tolist() == list(range(8))
    # Blocks of 3: starting at 0, 1, 2 stay in the first run; 5 is the only start in the second
    assert block_starts(timestamps, 3).tolist() == [0, 1, 2, 5]
    assert block_starts(timestamps, 6).tolist() == []


def test_block_bootstrap_shape_index_and_start_price():
    returns = np.log([1.01, 0.99, 1.02, 0.98, 1.0])
    prices = generate_block_bootstrap_price_series(
        returns, np.arange(4), n_prices=8, block_length=2, start_price=40_000, time_step="100ms", seed=0
    )
    assert len(prices) == 8
    assert prices.iloc[0] == pytest.approx(40_000)
    assert isinstance(prices.index, pd.TimedeltaIndex)
    assert (prices.index.to_series().diff().dropna() == pd.Timedelta("100ms")).all()


def test_block_bootstrap_keeps_blocks_intact():
    # Distinct returns, so every step of the path identifies the historical return it came from
    returns = np.arange(1, 21) * 1e-4
    prices = generate_block_bootstrap_price_series(returns, np.arange(16), n_prices=21, block_length=5, seed=3)
    sampled = np.diff(np.log(prices.to_numpy()))
    positions = np.rint(sampled / 1e-4).astype(int) - 1
    for block in positions.reshape(4, 5):
        assert (np.diff(block) == 1).all()


def test_block_bootstrap_only_uses_allowed_starts():
    returns = np.arange(1, 21) * 1e-4
    prices = generate_block_bootstrap_price_series(returns, [7], n_prices=8, block_length=3, seed=1)
    sampled = np.diff(np.log(prices.to_numpy()))
    # Two full blocks and one cut short, all starting at position 7
    assert np.rint(sampled / 1e-4).astype(int).tolist() == [8, 9, 10, 8, 9, 10, 8]


def test_block_bootstrap_seed_makes_the_path_reproducible():
    returns = np.random.default_rng(0).normal(0, 1e-3, 100)
    kwargs = dict(starts=np.arange(91), n_prices=50, block_length=10)
    first = generate_block_bootstrap_price_series(returns, seed=7, **kwargs)
    second = generate_block_bootstrap_price_series(returns, seed=7, **kwargs)
    other = generate_block_bootstrap_price_series(returns, seed=8, **kwargs)
    pd.testing.assert_series_equal(first, second)
    assert not np.allclose(first, other)
