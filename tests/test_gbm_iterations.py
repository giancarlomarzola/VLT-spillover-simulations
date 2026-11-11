import pandas as pd
import pytest

from core.gbm_iterations import run_iteration, saved_seeds, seed_path, simulate_seed
from core.price_process import generate_gbm_price_series
from core.simulation import run_simulation, summarise_run


@pytest.fixture
def setup(up_product, orderbook):
    """A small iteration: two volatilities, with and without an orderbook."""
    return {
        "vletps": up_product,
        "orderbooks": [None, orderbook],
        "volatilities": [0.5, 5.0],
        "n_prices": 500,
        "start_price": 100,
        "mu": 0.0,
        "time_step": "1h",
    }


def test_simulate_seed_has_one_row_per_volatility_and_orderbook(setup):
    summary, results = simulate_seed(7, **setup)
    assert list(summary["price_series"]) == ["vol_50", "vol_50", "vol_500", "vol_500"]
    assert list(summary["orderbook"]) == ["no_orderbook", "book", "no_orderbook", "book"]
    assert (summary["seed"] == 7).all()
    assert results == {}


def test_simulate_seed_matches_a_direct_run(setup, up_product, orderbook):
    summary, results = simulate_seed(7, **setup, keep_results=True)
    prices = generate_gbm_price_series(n_prices=500, start_price=100, mu=0.0, sigma=5.0, time_step="1h", seed=7)
    expected = run_simulation(up_product, orderbook, prices)
    pd.testing.assert_frame_equal(results["vol_500", "book"], expected)
    row = summary.set_index(["price_series", "orderbook"]).loc[("vol_500", "book")]
    assert row["n_rebalances"] == summarise_run(expected, up_product)["n_rebalances"] > 0


def test_simulate_seed_is_reproducible_and_seed_dependent(setup):
    first, _ = simulate_seed(1, **setup)
    again, _ = simulate_seed(1, **setup)
    other, _ = simulate_seed(2, **setup)
    pd.testing.assert_frame_equal(first, again)
    assert not first["realised_vol_raw"].equals(other["realised_vol_raw"])


def test_wipeout_times_are_elapsed_times_even_without_a_wipeout(setup):
    setup["volatilities"] = [0.01]  # too calm to wipe the product out
    summary, _ = simulate_seed(0, **setup)
    assert summary["wipeout_time_UP"].isna().all()
    assert summary["wipeout_time_UP"].dtype.kind == "m"


def test_run_iteration_saves_the_summary(setup, tmp_path):
    seed, seconds = run_iteration(3, tmp_path, **setup)
    assert seed == 3
    assert seconds > 0
    assert saved_seeds(tmp_path) == {3}
    assert [path.name for path in tmp_path.iterdir()] == ["seed_000003.parquet"]  # no temporary file left
    expected, _ = simulate_seed(3, **setup)
    pd.testing.assert_frame_equal(pd.read_parquet(seed_path(tmp_path, 3)), expected, check_dtype=False)


def test_saved_seeds_is_empty_for_an_empty_folder(tmp_path):
    assert saved_seeds(tmp_path) == set()
