import numpy as np
import pandas as pd
import pytest

from core.price_process import SECONDS_PER_YEAR, generate_gbm_price_series


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
