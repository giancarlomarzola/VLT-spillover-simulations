"""
Building blocks for experiments: the products and orderbooks they run on, one set per market.

    TSLA: products sized in line with the total for products on TSLA/NVDA etc., on a deep orderbook
        (used for the GBM simulations).
    BTC:  products sized like the real BTCUP/BTCDOWN on 2021-05-19, on a BTC-sized orderbook
        (used for the BTC bootstrap simulations).

Defined once here and picked by name in the simulations/run_*_experiment.py scripts (and reused by the plotting
scripts), so every experiment uses the same definitions. Changing a definition here does not
silently mix results: an experiment folder rejects a setup that differs from its config.json.
"""

import pandas as pd

from core.orderbook import Orderbook
from core.variable_leverage_product import VLETP

# ========================================
# Shared building blocks
# ========================================

def vletp_pair(
    up_name: str,
    down_name: str,
    up_nav: float,
    down_nav: float,
    start_leverage: float,
    leverage_lower_bound: float,
    leverage_upper_bound: float,
) -> list[VLETP]:
    """
    An UP and a DOWN product with the given NAVs, rebalancing to the nearest boundary of the band.
    start_leverage and the band are given as positive magnitudes; the DOWN product gets them mirrored
    below zero.
    """
    up = VLETP(
        name=up_name,
        start_nav=up_nav,
        start_leverage=start_leverage,
        leverage_lower_bound=leverage_lower_bound,
        leverage_upper_bound=leverage_upper_bound,
        leverage_target=None,  # None: nearest boundary; number: fixed target; (lower, upper): nearest of the two
    )
    down = VLETP(
        name=down_name,
        start_nav=down_nav,
        start_leverage=-start_leverage,
        leverage_lower_bound=-leverage_lower_bound,
        leverage_upper_bound=-leverage_upper_bound,
        leverage_target=None,
    )
    return [up, down]


def orderbook_sensitivities(baseline: dict) -> list[Orderbook | None]:
    """
    The reference run without an orderbook (None: rebalances have no price impact), the `baseline`
    book (Orderbook parameters), and the baseline with one parameter changed at a time.
    """

    def book(name: str, **changes) -> Orderbook:
        return Orderbook(name=name, **{**baseline, **changes})

    return [
        None,
        book("baseline"),
        book("depth 0.5x", depth=0.5 * baseline["depth"]),
        book("width 2x", width=2 * baseline["width"]),
        book("spread 2x", spread=2 * baseline["spread"]),
        book("shape k1", k=1),
        book("shape k1_5", k=1.5),
        book("resilience 30min", resilience=pd.Timedelta("30min")),
    ]


# ========================================
# TSLA (GBM simulations)
# ========================================

TSLA_BASELINE_BOOK = {
    "depth": 250_000_000,
    "width": 0.01,
    "spread": 0.5 / 10_000,
    "k": 0.5,
    "resilience": pd.Timedelta("5min"),  # half-life of orderbook depth recovery; 0=instant
}


def tsla_vletps(asset: str = "TSLA") -> list[VLETP]:
    """
    The UP and DOWN product (names "<asset>_UP" and "<asset>_DOWN"), sized in line with the total for
    products on TSLA/NVDA etc. `asset` only sets the names, e.g. "GBM" for the GBM simulations.
    """
    return vletp_pair(
        f"{asset}_UP",
        f"{asset}_DOWN",
        up_nav=3_000_000_000,
        down_nav=100_000_000,
        start_leverage=2,
        leverage_lower_bound=1.25,
        leverage_upper_bound=3,
    )


def tsla_orderbooks() -> list[Orderbook | None]:
    """orderbook_sensitivities around TSLA_BASELINE_BOOK."""
    return orderbook_sensitivities(TSLA_BASELINE_BOOK)


# ========================================
# BTC (bootstrap simulations)
# ========================================

BTC_BASELINE_BOOK = {
    "depth": 50_000_000,
    "width": 0.01,
    "spread": 0.5 / 10_000,
    "k": 0.5,
    "resilience": pd.Timedelta("5min"),  # half-life of orderbook depth recovery; 0=instant
}


def btc_vletps() -> list[VLETP]:
    """
    BTCUP and BTCDOWN, with NAVs approximating the real products on 2021-05-19 (about 37M and 19M at
    the start of the data), on the band of the real Binance leveraged tokens (1.25-4), but starting
    at leverage +/-2.5 (roughly the middle of the band) instead of the recorded leverage.
    """
    return vletp_pair(
        "BTCUP",
        "BTCDOWN",
        up_nav=40_000_000,
        down_nav=20_000_000,
        start_leverage=2.5,
        leverage_lower_bound=1.25,
        leverage_upper_bound=4,
    )


def btc_orderbooks() -> list[Orderbook | None]:
    """orderbook_sensitivities around BTC_BASELINE_BOOK."""
    return orderbook_sensitivities(BTC_BASELINE_BOOK)
