import math

import pandas as pd


def _replenish_seconds(value):
    """Convert a resilience value (duration string, Timedelta, seconds or None) to seconds."""
    if value is None:
        return math.inf
    seconds = pd.Timedelta(value).total_seconds() if isinstance(value, (str, pd.Timedelta)) else float(value)
    if seconds < 0:
        raise ValueError(f"resilience must not be negative, got {value!r}")
    return seconds


def _bid_ask(value, param):
    """Split a parameter into (bid, ask): a single number applies to both sides."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{param} must be a single number or a (bid, ask) pair, got {value!r}")
        return value[0], value[1]
    return value, value


class Orderbook:
    """
    Represents a limit orderbook, optionally asymmetric between bid and ask sides.

    Every parameter except name takes either a single number, used for both the bid and
    the ask side, or a (bid, ask) tuple of two numbers for an asymmetric book, e.g.
    depth=50_000_000 or depth=(30_000_000, 50_000_000).

    Parameters:
        name: Human-readable name for this orderbook.
        depth: Market depth in USDT.
        width: Maximum slippage at full depth (in decimal, e.g., 0.01 for 1%).
        spread: Spread at zero depth (in decimal, e.g., 0.005 for 0.5%).
        k: Curvature parameter for slippage formula.
            k=1 gives linear slippage; k<1 gives convex curve; k>1 gives concave curve
        resilience: Time needed to fully replenish an empty book, as a duration string or
            Timedelta (e.g. "30min", "2h") or a number of seconds. Depth recovers linearly,
            so half that time restores half of full depth. 0=replenished instantly
            (perfect resilience, the default), None=depth never recovers.

    The per-side values are stored as <param>_bid and <param>_ask (e.g. width_bid, width_ask); 
    resilience_bid/resilience_ask are stored in seconds. 
    depth_bid/depth_ask track the current, possibly depleted, depth;
    depth_bid_full/depth_ask_full keep the original depth.
    """

    def __init__(
        self,
        name: str,
        depth: float | tuple[float, float],
        width: float | tuple[float, float],
        spread: float | tuple[float, float],
        k: float | tuple[float, float] = 1,
        resilience: str | float | pd.Timedelta | None | tuple = 0,
    ):
        if not isinstance(name, str):
            raise TypeError(f"name must be string, got {type(name)}")

        self.name = name
        self.depth_bid_full, self.depth_ask_full = _bid_ask(depth, "depth")  # Original depth
        self.depth_bid, self.depth_ask = self.depth_bid_full, self.depth_ask_full  # Current depth (can be depleted)
        self.width_bid, self.width_ask = _bid_ask(width, "width")
        self.spread_bid, self.spread_ask = _bid_ask(spread, "spread")
        self.k_bid, self.k_ask = _bid_ask(k, "k")
        resilience_bid, resilience_ask = _bid_ask(resilience, "resilience")
        # Seconds needed to fully replenish each side (inf = never)
        self.resilience_bid = _replenish_seconds(resilience_bid)
        self.resilience_ask = _replenish_seconds(resilience_ask)

    def reset(self):
        """Restore both sides to full depth, undoing any depletion from earlier trades."""
        self.depth_bid, self.depth_ask = self.depth_bid_full, self.depth_ask_full

    def _log_parameters(self):
        """Log orderbook parameters for verification."""
        print(f"\n{'=' * 60}")
        print(f"Orderbook: {self.name}")
        print(f"{'=' * 60}")
        print(f"  Depth (tokens):    bid={self.depth_bid_full:>15,.0f}  ask={self.depth_ask_full:>15,.0f}")
        print(f"  Width (decimal):   bid={self.width_bid:>15.6f}  ask={self.width_ask:>15.6f}")
        print(f"  Spread (decimal):  bid={self.spread_bid:>15.6f}  ask={self.spread_ask:>15.6f}")
        print(f"  Curvature (k):     bid={self.k_bid:>15.6f}  ask={self.k_ask:>15.6f}")
        print(f"  Resilience (s):    bid={self.resilience_bid:>15,.1f}  ask={self.resilience_ask:>15,.1f}")
        print(f"{'=' * 60}\n")

    def execute_transaction(self, token_amount, effective_price):
        """
        Calculate signed slippage and execution scale for a trade of token_amount at the
        given effective price, by inverting the synthetic orderbook f(z) = D * t/(k + (1-k)*t).
        Automatically selects bid side (d < 0) or ask side (d > 0) based on trade direction.
        Depletes the orderbook depth by the executed amount.

        Returns: (slippage, scale) tuple where scale is the execution cap ratio.
        """
        if token_amount == 0:
            return 0.0, 1.0

        if token_amount > 0:
            D = self.depth_ask / effective_price
            S = self.spread_ask
            W = self.width_ask
            K = self.k_ask
            side = "ask"
        else:
            D = self.depth_bid / effective_price
            S = self.spread_bid
            W = self.width_bid
            K = self.k_bid
            side = "bid"

        x = abs(token_amount)
        actual_execution = min(D, x)  # Amount actually executed given available depth

        if D == 0 or x > D:
            s = W
            scale = D / x if x > 0 else 1.0
        else:
            # Slippage formula: use full depth D, not D_remaining
            s = S + (W - S) * (K * x) / (D - (1 - K) * x)
            scale = 1.0

        # Deplete the orderbook by converting back to USD and updating tracked depth
        if side == "ask":
            self.depth_ask -= actual_execution * effective_price
        else:
            self.depth_bid -= actual_execution * effective_price

        slippage = s if token_amount > 0 else -s
        return slippage, scale

    def replenish(self, seconds_elapsed):
        """
        Replenish orderbook depth on both sides for the time that has passed.

        Each side regains the fraction seconds_elapsed / resilience of its full depth
        (resilience being the time needed to refill completely), capped at full depth.

        Args:
            seconds_elapsed: Time since the previous replenishment, in seconds
        """
        fraction_restored_bid = self._fraction_restored(seconds_elapsed, self.resilience_bid)
        fraction_restored_ask = self._fraction_restored(seconds_elapsed, self.resilience_ask)

        self.depth_bid = min(self.depth_bid + fraction_restored_bid * self.depth_bid_full, self.depth_bid_full)
        self.depth_ask = min(self.depth_ask + fraction_restored_ask * self.depth_ask_full, self.depth_ask_full)

    @staticmethod
    def _fraction_restored(seconds_elapsed, seconds_to_full):
        """Fraction of full depth restored in seconds_elapsed, between 0 and 1."""
        if seconds_to_full == 0:
            return 1.0  # instant replenishment
        return min(seconds_elapsed / seconds_to_full, 1.0)

    def __repr__(self):
        k_part = "" if (self.k_bid == 1 and self.k_ask == 1) else f", k_bid={self.k_bid}, k_ask={self.k_ask}"
        resilience_part = (
            ""
            if (self.resilience_bid == 0 and self.resilience_ask == 0)
            else f", resilience_bid={self.resilience_bid}s, resilience_ask={self.resilience_ask}s"
        )

        return (
            f"Orderbook({self.name!r}, depth_bid={self.depth_bid}, depth_ask={self.depth_ask}, "
            f"width_bid={self.width_bid * 100}, width_ask={self.width_ask * 100}, "
            f"spread_bid={self.spread_bid * 100}, spread_ask={self.spread_ask * 100}"
            f"{k_part}{resilience_part})"
        )
