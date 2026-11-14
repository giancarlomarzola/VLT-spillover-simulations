import numpy as np
import pandas as pd
import pytest

from core.bootstrap_price_generator import (
    BootstrapPriceGenerator,
    block_starts,
    generate_block_bootstrap_price_series,
)


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


def test_generator_resamples_the_saved_returns(tmp_path):
    index = pd.date_range("2021-05-19", periods=1_000, freq="100ms", tz="UTC")
    returns = pd.Series(np.random.default_rng(0).normal(0, 1e-4, len(index)), index=index, name="log_return")
    returns.to_frame().to_parquet(tmp_path / "returns.parquet")
    generator = BootstrapPriceGenerator(
        currency="btc", time_step="100ms", n_prices=101, block="1s", start_price=40_000,
        returns_path=tmp_path / "returns.parquet",
    )
    generator.prepare()  # the file exists, so nothing is rebuilt
    [(label, prices)] = generator.paths(seed=1)
    assert label == "btc_bootstrap_1s"
    assert len(prices) == generator.n_prices == 101
    assert prices.iloc[0] == pytest.approx(40_000)
    pd.testing.assert_series_equal(prices, generator.paths(seed=1)[0][1])
