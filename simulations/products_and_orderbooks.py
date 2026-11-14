"""
Building blocks for experiments: the products and orderbooks they run on.

Defined once here and picked by name in the simulations/run_*_experiment.py scripts (and reused by the plotting
scripts), so every experiment uses the same definitions. Changing a definition here does not
silently mix results: an experiment folder rejects a setup that differs from its config.json.
"""

import pandas as pd

from core.orderbook import Orderbook
from core.variable_leverage_product import VLETP

# ========================================
# Products
# ========================================


def vletp_pair(asset: str) -> list[VLETP]:
    """
    The UP and DOWN product on `asset` (names "<asset>_UP" and "<asset>_DOWN"), sized in line with the
    total for products on TSLA/NVDA etc.
    """
    up = VLETP(
        name=f"{asset}_UP",
        start_nav=3_000_000_000,
        start_leverage=2,
        leverage_lower_bound=1.25,
        leverage_upper_bound=3,
        leverage_target=None,  # None: nearest boundary; number: fixed target; (lower, upper): nearest of the two
    )
    down = VLETP(
        name=f"{asset}_DOWN",
        start_nav=100_000_000,
        start_leverage=-2,
        leverage_lower_bound=-1.25,
        leverage_upper_bound=-3,
        leverage_target=None,
    )
    return [up, down]


# ========================================
# Orderbooks
# ========================================

BASELINE_BOOK = {
    "depth": 250_000_000,
    "width": 0.01,
    "spread": 0.005 / 100,
    "k": 0.5,
    "resilience": pd.Timedelta("5min"),  # half-life of orderbook depth recovery; 0=instant
}


def book(name: str, **changes) -> Orderbook:
    """An orderbook with the baseline parameters, except for `changes` (e.g. book("width 2x", width=0.02))."""
    return Orderbook(name=name, **{**BASELINE_BOOK, **changes})


def orderbook_sensitivities() -> list[Orderbook | None]:
    """
    The reference run without an orderbook (None: rebalances have no price impact), the baseline
    book, and the baseline with one parameter changed at a time.
    """
    b = BASELINE_BOOK
    return [
        None,
        book("baseline"),
        book("depth 0.5x", depth=0.5 * b["depth"]),
        book("width 2x", width=2 * b["width"]),
        book("spread 2x", spread=2 * b["spread"]),
        book("shape k1", k=1),
        book("shape k1_5", k=1.5),
        book("resilience 30min", resilience=pd.Timedelta("30min")),
    ]
