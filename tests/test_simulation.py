import numpy as np
import pandas as pd
import pytest

from core.orderbook import Orderbook
from core.price_process import generate_gbm_price_series
from core.simulation import MARKET_COLUMNS, PRODUCT_VARS, _longest_run, run_simulation, summarise_run
from core.variable_leverage_product import VLETP

# A 40% drop takes the UP fixture (NAV 100, leverage 2, band [1, 3]) to NAV 20, exposure 120,
# leverage 6, so it sells 60 to get back to leverage 3. The DOWN fixture goes to NAV 180,
# exposure -120, leverage 0.667, so it also sells 60 to get back to leverage 1.
DROP = [100, 100, 60]


# --- input validation ---------------------------------------------------------------


def test_duplicate_names_rejected(up_product, price_series):
    twin = VLETP("UP", 100, 2, 1, 3)
    with pytest.raises(ValueError, match="unique"):
        run_simulation([up_product, twin], None, price_series(DROP))


def test_no_products_rejected(price_series):
    with pytest.raises(ValueError, match="at least one"):
        run_simulation([], None, price_series(DROP))


def test_prices_must_be_a_series(up_product):
    with pytest.raises(TypeError, match="Series"):
        run_simulation(up_product, None, DROP)


def test_prices_need_a_time_index(up_product):
    with pytest.raises(TypeError, match="Index"):
        run_simulation(up_product, None, pd.Series(DROP))


def test_prices_must_be_sorted_in_time(up_product, price_series):
    with pytest.raises(ValueError, match="sorted"):
        run_simulation(up_product, None, price_series(DROP).iloc[::-1])


def test_empty_prices_rejected(up_product, price_series):
    with pytest.raises(ValueError, match="empty"):
        run_simulation(up_product, None, price_series([]))


# --- output shape -------------------------------------------------------------------


def test_output_columns_and_index(up_product, down_product, orderbook, price_series):
    prices = price_series(DROP)
    result = run_simulation([up_product, down_product], orderbook, prices)
    expected = MARKET_COLUMNS + [f"{prefix}_{name}" for name in ("UP", "DOWN") for prefix in PRODUCT_VARS.values()]
    assert list(result.columns) == expected
    assert result.index.equals(prices.index)


def test_datetime_index_accepted(up_product, orderbook):
    index = pd.date_range("2024-01-01", periods=3, freq="1min")
    result = run_simulation(up_product, orderbook, pd.Series(DROP, index=index, dtype=float))
    assert result.index.equals(index)
    assert result["actual_total_delta"].iloc[2] == pytest.approx(-60)


def test_first_row_is_the_initial_state(up_product, orderbook, price_series):
    first = run_simulation(up_product, orderbook, price_series(DROP)).iloc[0]
    assert first["raw_price"] == 100
    assert first["simulated_price"] == 100
    assert first["price_multiplier"] == 1
    assert first["nav_UP"] == 100
    assert first["exposure_UP"] == 200
    assert first["leverage_UP"] == 2
    assert first["actual_total_delta"] == 0
    assert first["depth_bid"] == 1_000_000


def test_single_price_gives_one_row(up_product, price_series):
    result = run_simulation(up_product, None, price_series([100]))
    assert len(result) == 1
    assert result["nav_UP"].iloc[0] == 100


# --- behaviour ----------------------------------------------------------------------


def test_flat_prices_cause_no_rebalancing(up_product, down_product, orderbook, price_series):
    result = run_simulation([up_product, down_product], orderbook, price_series([100] * 6))
    assert (result["actual_total_delta"] == 0).all()
    assert (result["price_multiplier"] == 1).all()
    assert (result["nav_UP"] == 100).all()
    assert (result["nav_DOWN"] == 100).all()
    assert (result["depth_bid"] == 1_000_000).all()


def test_start_leverage_outside_band_rebalances_on_first_step(orderbook, price_series):
    # Leverage 5 in a band of [1, 3] on a flat price: sells 200 at the first step to reach 3
    product = VLETP("UP", 100, 5, 1, 3)
    result = run_simulation(product, orderbook, price_series([100, 100, 100]))
    assert result["leverage_UP"].iloc[0] == 5  # initial row shows the position as given
    assert result["exposure_UP"].iloc[0] == 500
    assert result["actual_total_delta"].iloc[0] == 0
    assert result["actual_total_delta"].iloc[1] == pytest.approx(-200)
    assert result["leverage_UP"].iloc[1] == pytest.approx(5)
    assert result["leverage_after_UP"].iloc[1] == pytest.approx(3)
    assert result["orderbook_effect"].iloc[1] < 0


def test_without_orderbook_price_is_not_impacted(up_product, price_series):
    result = run_simulation(up_product, None, price_series(DROP))
    assert (result["simulated_price"] == result["raw_price"]).all()
    assert (result["price_multiplier"] == 1).all()
    assert (result["orderbook_effect"] == 0).all()
    assert result["depth_bid"].isna().all()
    assert result["depth_ask"].isna().all()


def test_price_drop_triggers_expected_rebalance(up_product, price_series):
    result = run_simulation(up_product, None, price_series(DROP))
    assert result["actual_total_delta"].iloc[1] == 0  # flat step, no trade
    row = result.iloc[2]
    assert row["nav_UP"] == pytest.approx(20)
    assert row["exposure_UP"] == pytest.approx(120)
    assert row["leverage_UP"] == pytest.approx(6)
    assert row["target_delta_UP"] == pytest.approx(-60)
    assert row["actual_delta_UP"] == pytest.approx(-60)
    assert row["exposure_after_UP"] == pytest.approx(60)
    assert row["leverage_after_UP"] == pytest.approx(3)
    assert row["target_total_delta"] == pytest.approx(-60)
    assert row["actual_total_delta"] == pytest.approx(-60)


def test_trades_of_all_products_are_summed(up_product, down_product, price_series):
    row = run_simulation([up_product, down_product], None, price_series(DROP)).iloc[2]
    assert row["target_delta_UP"] == pytest.approx(-60)
    assert row["target_delta_DOWN"] == pytest.approx(-60)
    assert row["leverage_after_DOWN"] == pytest.approx(1)
    assert row["target_total_delta"] == pytest.approx(-120)
    assert row["actual_total_delta"] == pytest.approx(-120)


def test_rebalance_slippage_and_depth(up_product, orderbook, price_series):
    # Selling 60 into a 1,000,000 deep bid: spread + (width - spread) * 60 / 1,000,000
    slippage = -(0.001 + 0.009 * 60 / 1_000_000)
    row = run_simulation(up_product, orderbook, price_series(DROP)).iloc[2]
    assert row["orderbook_effect"] == pytest.approx(slippage)
    assert row["price_multiplier"] == pytest.approx(1 + slippage)
    assert row["simulated_price"] == pytest.approx(60 * (1 + slippage))
    assert row["raw_price"] == 60
    assert row["depth_bid"] == pytest.approx(1_000_000 - 60)  # recorded before replenishment
    assert row["depth_ask"] == 1_000_000


def test_price_impact_reaches_the_products_one_step_later(orderbook, price_series):
    # Fixed target of 2, so the small impact return at the last step does not trigger a new trade
    product = VLETP("UP", 100, 2, 1, 3, leverage_target=2)
    result = run_simulation(product, orderbook, price_series([100, 100, 60, 60]))
    slippage = result["orderbook_effect"].iloc[2]
    assert slippage < 0

    # The step of the trade itself is marked to market on the raw return only
    assert result["nav_UP"].iloc[2] == pytest.approx(20)
    # Next step: the raw price is flat, so the only return is the impact of the earlier trade
    assert result["nav_UP"].iloc[3] == pytest.approx(20 * (1 + 2 * slippage))
    assert result["exposure_UP"].iloc[3] == pytest.approx(40 * (1 + slippage))
    assert result["actual_total_delta"].iloc[3] == 0
    assert result["price_multiplier"].iloc[3] == pytest.approx(1 + slippage)  # impact is permanent


def test_impact_accumulates_across_rebalances(up_product, orderbook, price_series):
    # After the first drop leverage sits on the upper boundary, so each further 10% drop breaches it again
    result = run_simulation(up_product, orderbook, price_series([100, 60, 54, 48.6]))
    effects = result["orderbook_effect"].to_numpy()
    assert (effects[1:] < 0).all()
    assert result["price_multiplier"].to_numpy() == pytest.approx(np.cumprod(1 + effects))


def test_trade_is_scaled_back_when_depth_runs_out(up_product, price_series):
    # Only 30 of bid depth for a trade of 60: half fills, at the full width
    thin = Orderbook("thin", depth=30, width=0.01, spread=0.001)
    row = run_simulation(up_product, thin, price_series(DROP)).iloc[2]
    assert row["target_total_delta"] == pytest.approx(-60)
    assert row["actual_total_delta"] == pytest.approx(-30)
    assert row["actual_delta_UP"] == pytest.approx(-30)
    assert row["leverage_after_UP"] == pytest.approx(4.5)  # still outside the band
    assert row["orderbook_effect"] == pytest.approx(-0.01)
    assert row["depth_bid"] == pytest.approx(0)


def test_all_products_are_scaled_back_equally(up_product, down_product, price_series):
    thin = Orderbook("thin", depth=30, width=0.01, spread=0.001)  # 30 available for 120 wanted
    row = run_simulation([up_product, down_product], thin, price_series(DROP)).iloc[2]
    assert row["actual_delta_UP"] == pytest.approx(-15)
    assert row["actual_delta_DOWN"] == pytest.approx(-15)
    assert row["actual_total_delta"] == pytest.approx(-30)


def test_orderbook_replenishes_between_steps(up_product, price_series):
    # 0.1s half-life and 1s steps: ten half-lives pass per step, so the gap shrinks ~1000x
    book = Orderbook("book", depth=1_000, width=0.01, spread=0.001, resilience="0.1s")
    result = run_simulation(up_product, book, price_series([100, 100, 60, 60]))
    assert result["depth_bid"].iloc[2] == pytest.approx(940)
    assert result["depth_bid"].iloc[3] == pytest.approx(1_000, abs=1)  # refilled, minus a tiny follow-up trade


def test_wipe_out_liquidates_and_stays_dead(up_product, price_series):
    result = run_simulation(up_product, None, price_series([100, 50, 80]))
    assert result["nav_UP"].tolist() == [100, 0, 0]
    assert result["actual_total_delta"].iloc[1] == pytest.approx(-100)  # sells the remaining exposure
    assert result["actual_total_delta"].iloc[2] == 0
    assert result["exposure_after_UP"].iloc[2] == 0


def test_objects_can_be_reused_across_runs(up_product, down_product):
    book = Orderbook("book", depth=5_000, width=0.01, spread=0.001, resilience=None)
    prices = generate_gbm_price_series(500, sigma=30, seed=3)
    first = run_simulation([up_product, down_product], book, prices)
    assert (first["actual_total_delta"] != 0).any()  # the path does make the products trade
    assert book.depth_bid < 5_000 or book.depth_ask < 5_000  # objects are left in their end state
    second = run_simulation([up_product, down_product], book, prices)
    pd.testing.assert_frame_equal(first, second)


# --- summarise_run ------------------------------------------------------------------


@pytest.mark.parametrize(
    "mask, expected",
    [
        ([], 0),
        ([False, False], 0),
        ([True], 1),
        ([True, True, False, True, True, True], 3),
        ([True, True, True], 3),
        ([False, True, False, True], 1),
    ],
)
def test_longest_run(mask, expected):
    assert _longest_run(np.array(mask, dtype=bool)) == expected


def test_summary_of_a_single_filled_rebalance(up_product, orderbook, price_series):
    slippage = -(0.001 + 0.009 * 60 / 1_000_000)
    result = run_simulation(up_product, orderbook, price_series(DROP))
    summary = summarise_run(result, up_product)

    assert summary["n_rebalances"] == 1
    assert summary["longest_rebalance_run"] == 1
    assert summary["avg_rebalance_size"] == pytest.approx(60)
    assert summary["total_abs_rebalance_size"] == pytest.approx(60)
    assert summary["fill_ratio"] == pytest.approx(1)
    assert summary["avg_negative_impact"] == pytest.approx(slippage)
    assert summary["max_negative_impact"] == pytest.approx(slippage)
    assert summary["avg_abs_impact"] == pytest.approx(-slippage)
    assert np.isnan(summary["avg_positive_impact"])  # no buys to average over
    assert np.isnan(summary["max_positive_impact"])
    assert summary["final_price_deviation"] == pytest.approx(slippage)
    assert summary["min_price_deviation"] == pytest.approx(slippage)
    assert summary["max_price_deviation"] == 0
    assert summary["min_depth_bid"] == pytest.approx(1_000_000 - 60)
    assert summary["min_depth_ask"] == 1_000_000
    assert summary["final_nav_UP"] == pytest.approx(20)
    assert pd.isna(summary["wipeout_time_UP"])
    assert summary["n_outside_band_UP"] == 0


def test_summary_reports_unfilled_rebalances(up_product, price_series):
    thin = Orderbook("thin", depth=30, width=0.01, spread=0.001)
    result = run_simulation(up_product, thin, price_series(DROP))
    summary = summarise_run(result, [up_product])
    assert summary["fill_ratio"] == pytest.approx(0.5)
    assert summary["n_outside_band_UP"] == 1
    assert summary["min_depth_bid"] == pytest.approx(0)


def test_summary_reports_wipe_out_time(up_product, down_product, price_series):
    prices = price_series([100, 50, 80])
    summary = summarise_run(run_simulation([up_product, down_product], None, prices), [up_product, down_product])
    assert summary["wipeout_time_UP"] == prices.index[1]
    assert summary["final_nav_UP"] == 0
    assert summary["n_outside_band_UP"] == 0  # a dead product is not counted as outside its band
    assert pd.isna(summary["wipeout_time_DOWN"])
    assert summary["final_nav_DOWN"] > 0


def test_summary_without_any_rebalance(up_product, price_series):
    summary = summarise_run(run_simulation(up_product, None, price_series([100, 101, 100])), up_product)
    assert summary["n_rebalances"] == 0
    assert summary["longest_rebalance_run"] == 0
    assert np.isnan(summary["fill_ratio"])
    assert np.isnan(summary["avg_rebalance_size"])
    assert np.isnan(summary["min_depth_bid"])  # no orderbook
    assert summary["final_price_deviation"] == 0


def test_summary_volatility_ratio(up_product, down_product):
    prices = generate_gbm_price_series(2_000, sigma=0.5, time_step="1min", seed=5)
    untouched = summarise_run(run_simulation(up_product, None, prices), up_product)
    assert untouched["realised_vol_raw"] == pytest.approx(0.5, rel=0.1)
    assert untouched["realised_vol_ratio"] == pytest.approx(1)

    # A violent path through a thin book: rebalancing moves the price, so the volatilities differ
    wild = generate_gbm_price_series(2_000, sigma=30, seed=5)
    thin = Orderbook("thin", depth=2_000, width=0.05, spread=0.01)
    impacted = summarise_run(run_simulation([up_product, down_product], thin, wild), [up_product, down_product])
    assert impacted["n_rebalances"] > 0
    assert impacted["realised_vol_ratio"] != pytest.approx(1)
