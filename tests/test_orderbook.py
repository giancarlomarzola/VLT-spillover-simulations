import math

import pandas as pd
import pytest

from core.orderbook import Orderbook

PRICE = 100.0  # at this price the fixture book holds 10,000 tokens per side


# --- construction -----------------------------------------------------------------


def test_single_number_applies_to_both_sides():
    book = Orderbook("book", depth=500, width=0.02, spread=0.001, k=0.5, resilience=60)
    assert (book.depth_bid, book.depth_ask) == (500, 500)
    assert (book.depth_bid_full, book.depth_ask_full) == (500, 500)
    assert (book.width_bid, book.width_ask) == (0.02, 0.02)
    assert (book.spread_bid, book.spread_ask) == (0.001, 0.001)
    assert (book.k_bid, book.k_ask) == (0.5, 0.5)
    assert (book.resilience_bid, book.resilience_ask) == (60, 60)


def test_tuple_gives_asymmetric_book():
    book = Orderbook(
        "book", depth=(300, 500), width=(0.01, 0.02), spread=(0.001, 0.002), k=(0.5, 2), resilience=("1min", None)
    )
    assert (book.depth_bid, book.depth_ask) == (300, 500)
    assert (book.width_bid, book.width_ask) == (0.01, 0.02)
    assert (book.spread_bid, book.spread_ask) == (0.001, 0.002)
    assert (book.k_bid, book.k_ask) == (0.5, 2)
    assert (book.resilience_bid, book.resilience_ask) == (60, math.inf)


def test_defaults_are_linear_and_instantly_replenished():
    book = Orderbook("book", depth=500, width=0.02, spread=0.001)
    assert (book.k_bid, book.k_ask) == (1, 1)
    assert (book.resilience_bid, book.resilience_ask) == (0, 0)


@pytest.mark.parametrize("param", ["depth", "width", "spread", "k", "resilience"])
def test_wrong_length_tuple_rejected(param):
    kwargs = {"depth": 500, "width": 0.02, "spread": 0.001}
    kwargs[param] = (1, 2, 3)
    with pytest.raises(ValueError, match=param):
        Orderbook("book", **kwargs)


def test_name_must_be_string():
    with pytest.raises(TypeError):
        Orderbook(123, depth=500, width=0.02, spread=0.001)


@pytest.mark.parametrize(
    "resilience, expected_seconds",
    [
        ("30min", 1800),
        ("2h", 7200),
        (pd.Timedelta(seconds=45), 45),
        (90, 90),
        (0, 0),
        (None, math.inf),
    ],
)
def test_resilience_converted_to_seconds(resilience, expected_seconds):
    book = Orderbook("book", depth=500, width=0.02, spread=0.001, resilience=resilience)
    assert book.resilience_bid == expected_seconds
    assert book.resilience_ask == expected_seconds


@pytest.mark.parametrize("resilience", [-1, "-5s"])
def test_negative_resilience_rejected(resilience):
    with pytest.raises(ValueError, match="resilience"):
        Orderbook("book", depth=500, width=0.02, spread=0.001, resilience=resilience)


# --- execute_transaction ------------------------------------------------------------


def test_zero_trade_has_no_effect(orderbook):
    assert orderbook.execute_transaction(0, PRICE) == (0.0, 1.0)
    assert (orderbook.depth_bid, orderbook.depth_ask) == (1_000_000, 1_000_000)


def test_buy_hits_ask_with_positive_slippage(orderbook):
    # Half the ask (5,000 of 10,000 tokens): spread + half of (width - spread)
    slippage, scale = orderbook.execute_transaction(5_000, PRICE)
    assert slippage == pytest.approx(0.001 + 0.009 * 0.5)
    assert scale == 1.0
    assert orderbook.depth_ask == pytest.approx(500_000)
    assert orderbook.depth_bid == 1_000_000  # other side untouched


def test_sell_hits_bid_with_negative_slippage(orderbook):
    slippage, scale = orderbook.execute_transaction(-5_000, PRICE)
    assert slippage == pytest.approx(-(0.001 + 0.009 * 0.5))
    assert scale == 1.0
    assert orderbook.depth_bid == pytest.approx(500_000)
    assert orderbook.depth_ask == 1_000_000


def test_tiny_trade_pays_about_the_spread(orderbook):
    slippage, _ = orderbook.execute_transaction(1e-6, PRICE)
    assert slippage == pytest.approx(0.001)


def test_trade_for_exactly_full_depth_pays_the_width(orderbook):
    slippage, scale = orderbook.execute_transaction(10_000, PRICE)
    assert slippage == pytest.approx(0.01)
    assert scale == 1.0
    assert orderbook.depth_ask == pytest.approx(0)


def test_curvature_changes_slippage():
    # k = 0.5, half the depth: spread + (width - spread) * (0.5 * 5000) / (10000 - 0.5 * 5000)
    book = Orderbook("book", depth=1_000_000, width=0.01, spread=0.001, k=0.5)
    slippage, _ = book.execute_transaction(5_000, PRICE)
    assert slippage == pytest.approx(0.001 + 0.009 / 3)


def test_asymmetric_book_uses_the_side_being_hit():
    book = Orderbook("book", depth=(1_000_000, 2_000_000), width=(0.01, 0.02), spread=(0.001, 0.002))
    sell_slippage, _ = book.execute_transaction(-5_000, PRICE)  # half the bid
    buy_slippage, _ = book.execute_transaction(5_000, PRICE)  # a quarter of the ask
    assert sell_slippage == pytest.approx(-(0.001 + 0.009 * 0.5))
    assert buy_slippage == pytest.approx(0.002 + 0.018 * 0.25)


def test_trade_larger_than_depth_is_scaled_back(orderbook):
    # Wants 20,000 tokens, only 10,000 available
    slippage, scale = orderbook.execute_transaction(20_000, PRICE)
    assert slippage == pytest.approx(0.01)  # full width
    assert scale == pytest.approx(0.5)
    assert orderbook.depth_ask == pytest.approx(0)


def test_trade_against_empty_side_does_not_fill():
    book = Orderbook("book", depth=1_000_000, width=0.01, spread=0.001, resilience=None)
    book.execute_transaction(-20_000, PRICE)  # empty the bid
    slippage, scale = book.execute_transaction(-100, PRICE)
    assert slippage == pytest.approx(-0.01)
    assert scale == 0
    assert book.depth_bid == 0


def test_depletion_is_in_quote_currency(orderbook):
    # 1,000 tokens at a price of 50 removes 50,000 of depth
    orderbook.execute_transaction(1_000, 50.0)
    assert orderbook.depth_ask == pytest.approx(950_000)


# --- replenish / reset --------------------------------------------------------------


def test_replenish_is_linear_and_capped():
    book = Orderbook("book", depth=1_000_000, width=0.01, spread=0.001, resilience="100s")
    book.execute_transaction(10_000, PRICE)  # empty the ask
    assert book.depth_ask == pytest.approx(0)

    book.replenish(25)  # a quarter of the time restores a quarter of full depth
    assert book.depth_ask == pytest.approx(250_000)
    book.replenish(25)
    assert book.depth_ask == pytest.approx(500_000)
    book.replenish(1_000)  # never above full depth
    assert book.depth_ask == 1_000_000
    assert book.depth_bid == 1_000_000


def test_zero_resilience_replenishes_instantly(orderbook):
    orderbook.execute_transaction(10_000, PRICE)
    orderbook.replenish(0.001)
    assert orderbook.depth_ask == 1_000_000


def test_none_resilience_never_replenishes():
    book = Orderbook("book", depth=1_000_000, width=0.01, spread=0.001, resilience=None)
    book.execute_transaction(5_000, PRICE)
    book.replenish(1e9)
    assert book.depth_ask == pytest.approx(500_000)


def test_sides_replenish_at_their_own_rate():
    book = Orderbook("book", depth=1_000_000, width=0.01, spread=0.001, resilience=("100s", None))
    book.execute_transaction(-10_000, PRICE)
    book.execute_transaction(10_000, PRICE)
    book.replenish(50)
    assert book.depth_bid == pytest.approx(500_000)
    assert book.depth_ask == pytest.approx(0)


def test_reset_restores_full_depth():
    book = Orderbook("book", depth=(300_000, 500_000), width=0.01, spread=0.001, resilience=None)
    book.execute_transaction(-1_000, PRICE)
    book.execute_transaction(1_000, PRICE)
    book.reset()
    assert (book.depth_bid, book.depth_ask) == (300_000, 500_000)
