"""Shared fixtures. Numbers are chosen so expected values can be worked out by hand."""

import pandas as pd
import pytest

from core.orderbook import Orderbook
from core.variable_leverage_product import VLETP


@pytest.fixture
def up_product():
    """Long product: NAV 100, leverage 2 (exposure 200), band [1, 3], boundary rebalancing."""
    return VLETP("UP", start_nav=100, start_leverage=2, leverage_lower_bound=1, leverage_upper_bound=3)


@pytest.fixture
def down_product():
    """Short product: NAV 100, leverage -2 (exposure -200), band [-1, -3], boundary rebalancing."""
    return VLETP("DOWN", start_nav=100, start_leverage=-2, leverage_lower_bound=-1, leverage_upper_bound=-3)


@pytest.fixture
def orderbook():
    """Symmetric linear book: 1,000,000 deep per side, 0.1% spread, 1% width, instant replenishment."""
    return Orderbook("book", depth=1_000_000, width=0.01, spread=0.001)


@pytest.fixture
def price_series():
    """Factory turning a list of prices into a Series on a TimedeltaIndex with a fixed step."""

    def make(values, step="1s"):
        index = pd.timedelta_range(start=0, periods=len(values), freq=step, name="elapsed")
        return pd.Series(values, index=index, name="price", dtype=float)

    return make
