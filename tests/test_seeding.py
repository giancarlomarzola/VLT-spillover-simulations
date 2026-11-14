import json

import numpy as np
import pandas as pd

from core.experiment_runner import check_or_write_config, experiment_config, load_setup, simulate_seed
from core.gbm_price_generator import GBMPriceGenerator, generate_gbm_price_series
from core.seeding import iteration_seed


def test_without_a_master_seed_the_iteration_is_the_seed():
    assert iteration_seed(None, 42) == 42


def test_iteration_streams_are_the_spawned_children_of_the_master_seed():
    children = np.random.SeedSequence(123).spawn(5)
    for i, child in enumerate(children):
        expected = np.random.default_rng(child).random(10)
        assert np.array_equal(np.random.default_rng(iteration_seed(123, i)).random(10), expected)


def test_iteration_streams_depend_on_the_master_seed_and_the_iteration():
    draw = lambda master, i: np.random.default_rng(iteration_seed(master, i)).random(10)  # noqa: E731
    assert np.array_equal(draw(1, 3), draw(1, 3))
    assert not np.allclose(draw(1, 3), draw(1, 4))
    assert not np.allclose(draw(1, 3), draw(2, 3))


def make_gbm(master_seed=None):
    return GBMPriceGenerator(
        volatilities=[0.5], n_prices=50, start_price=100, mu=0.0, time_step="1h", master_seed=master_seed
    )


def test_gbm_generator_without_a_master_seed_draws_from_the_iteration_number():
    [(_, prices)] = make_gbm().paths(7)
    pd.testing.assert_series_equal(prices, generate_gbm_price_series(50, 100, 0.0, 0.5, "1h", seed=7))
    assert "master_seed" not in make_gbm().config()  # older config.json files still match


def test_gbm_generator_with_a_master_seed_draws_from_its_child():
    [(_, prices)] = make_gbm(master_seed=99).paths(7)
    expected = generate_gbm_price_series(50, 100, 0.0, 0.5, "1h", seed=np.random.SeedSequence(99).spawn(8)[7])
    pd.testing.assert_series_equal(prices, expected)
    assert make_gbm(master_seed=99).config()["master_seed"] == 99


def test_load_setup_restores_the_master_seed(tmp_path, up_product, orderbook):
    generator = make_gbm(master_seed=99)
    config = experiment_config(up_product, [orderbook], generator)
    check_or_write_config(tmp_path, config)
    assert json.loads((tmp_path / "config.json").read_text())["price_process"]["master_seed"] == 99
    setup = load_setup(tmp_path)
    assert setup["price_generator"].master_seed == 99
    original, _ = simulate_seed(5, up_product, [None, orderbook], generator)
    rebuilt, _ = simulate_seed(5, **setup)
    pd.testing.assert_frame_equal(original, rebuilt)
