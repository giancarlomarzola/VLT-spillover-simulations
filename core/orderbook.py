import math

import pandas as pd


# Relative gap below which a recovering side is snapped back to exactly full depth.
# Exponential recovery only approaches full depth asymptotically; snapping lets callers
# detect a full book with ==, at an error far below anything the simulation can resolve.
_FULL_DEPTH_TOLERANCE = 1e-12


def _replenish_seconds(value):
    """Convert a resilience half-life (duration string, Timedelta, seconds or None) to seconds."""
    if value is None:
        return math.inf
    seconds = pd.Timedelta(value).total_seconds() if isinstance(value, (str, pd.Timedelta)) else float(value)
    if seconds < 0:
        raise ValueError(f"resilience must not be negative, got {value!r}")
    return seconds


def _resilience_rate(half_life):
    """Resilience rate rho (per second) for a half-life in seconds: rho = ln(2) / half_life."""
    if half_life == 0:
        return math.inf  # instant replenishment
    return math.log(2) / half_life  # half_life = inf gives rho = 0, i.e. never recovers


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
        resilience: Half-life of depth recovery, as a duration string or Timedelta
            (e.g. "30min", "2h") or a number of seconds. Depth recovers exponentially:
            the missing depth halves every half-life, so one half-life restores 50% of
            what a trade removed, two restore 75%, three 87.5%. 0=replenished instantly
            (perfect resilience, the default), None=depth never recovers.

    The per-side values are stored as <param>_bid and <param>_ask (e.g. width_bid, width_ask);
    resilience_bid/resilience_ask hold the half-lives in seconds, and rho_bid/rho_ask the
    matching resilience rates rho = ln(2) / half-life, per second (inf = instant, 0 = never).
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
        # Half-life of each side's depth recovery, in seconds (inf = never)
        self.resilience_bid = _replenish_seconds(resilience_bid)
        self.resilience_ask = _replenish_seconds(resilience_ask)
        # Resilience rate of each side, per second
        self.rho_bid = _resilience_rate(self.resilience_bid)
        self.rho_ask = _resilience_rate(self.resilience_ask)

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
        print(f"  Half-life (s):     bid={self.resilience_bid:>15,.1f}  ask={self.resilience_ask:>15,.1f}")
        print(f"  Rho (1/s):         bid={self.rho_bid:>15.6g}  ask={self.rho_ask:>15.6g}")
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
        Exponentially replenish orderbook depth on both sides for the time that has passed.

        For each side s the missing depth decays at the resilience rate rho^s:

            d_{t+Δ} = d_0 - (d_0 - d_t) * exp(-rho * Δ),   rho = ln(2) / half-life

        Exponential recovery is memoryless, so calling this once per step gives the same
        depth as applying the formula once from the time of the last trade.

        Args:
            seconds_elapsed: Time since the previous replenishment, in seconds
        """
        self.depth_bid = self._replenish_side(self.depth_bid, self.depth_bid_full, seconds_elapsed, self.rho_bid)
        self.depth_ask = self._replenish_side(self.depth_ask, self.depth_ask_full, seconds_elapsed, self.rho_ask)

    @staticmethod
    def _replenish_side(depth, depth_full, seconds_elapsed, rho):
        """Depth of one side after seconds_elapsed of exponential recovery toward depth_full."""
        if seconds_elapsed <= 0 or rho == 0:
            return depth  # no time passed / never recovers
        if math.isinf(rho):
            return depth_full  # instant replenishment
        gap = (depth_full - depth) * math.exp(-rho * seconds_elapsed)  # remaining d_0 - d
        if gap <= _FULL_DEPTH_TOLERANCE * depth_full:
            return depth_full
        return depth_full - gap

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
