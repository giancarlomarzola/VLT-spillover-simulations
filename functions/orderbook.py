import numpy as np


class Orderbook:
    """
    Represents a limit orderbook with optional asymmetry between bid and ask sides.

    Parameters:
        name: Human-readable name for this orderbook.
        depth: Market depth in tokens. Scalar (symmetric) or (bid, ask) tuple (asymmetric).
        width: Maximum slippage at full depth in percent (e.g., 1 for 1%). Stored internally as decimal.
            Scalar (symmetric) or (bid, ask) tuple (asymmetric).
        spread: Spread at zero depth in percent (e.g., 0.5 for 0.5 bps). Stored internally as decimal.
            Scalar (symmetric) or (bid, ask) tuple (asymmetric).
        k: Curvature parameter for slippage formula. Scalar (symmetric) or (bid, ask) tuple (asymmetric).
            k=1 gives linear slippage; lower k gives more convex curve. Default 1.
        resilience: Resilience parameter (float between 0 and 1). 1=perfect resilience (full recovery at each step).
            Scalar (symmetric) or (bid, ask) tuple (asymmetric). Default 1.
    """

    def __init__(self, name, depth, width, spread, k=1, resilience=1):
        if not isinstance(name, str):
            raise TypeError(f"name must be string, got {type(name)}")
        if not isinstance(depth, (int, float, tuple, list)):
            raise TypeError(f"depth must be scalar or tuple/list, got {type(depth)}")
        if not isinstance(width, (int, float, tuple, list)):
            raise TypeError(f"width must be scalar or tuple/list, got {type(width)}")
        if not isinstance(spread, (int, float, tuple, list)):
            raise TypeError(f"spread must be scalar or tuple/list, got {type(spread)}")
        if not isinstance(k, (int, float, tuple, list)):
            raise TypeError(f"k must be scalar or tuple/list, got {type(k)}")
        if not isinstance(resilience, (int, float, tuple, list)):
            raise TypeError(f"resilience must be scalar or tuple/list, got {type(resilience)}")

        self.name = name
        self.depth = self._as_side_pair(depth)
        # Convert width and spread from percent to decimal (e.g., 1 -> 0.01, 0.5 -> 0.005)
        self.width = self._percent_to_decimal(self._as_side_pair(width))
        self.spread = self._percent_to_decimal(self._as_side_pair(spread))
        self.k = self._as_side_pair(k)
        self.resilience = self._as_side_pair(resilience)

    @staticmethod
    def _as_side_pair(param):
        """Convert scalar to symmetric (bid, ask) pair or validate tuple."""
        if np.isscalar(param):
            return (param, param)
        bid, ask = param
        return (bid, ask)

    @staticmethod
    def _percent_to_decimal(pair):
        """Convert percentage (bid, ask) pair to decimal representation."""
        return (pair[0] / 100, pair[1] / 100)

    def __repr__(self):
        def format_pair(pair):
            if pair[0] == pair[1]:
                return str(pair[0])
            return str(pair)

        def format_pair_percent(pair):
            percent_pair = (pair[0] * 100, pair[1] * 100)
            if percent_pair[0] == percent_pair[1]:
                return str(percent_pair[0])
            return str(percent_pair)

        depth_str = format_pair(self.depth)
        width_str = format_pair_percent(self.width)
        spread_str = format_pair_percent(self.spread)
        k_str = format_pair(self.k)
        resilience_str = format_pair(self.resilience)

        k_part = "" if self.k == (1, 1) else f", k={k_str}"
        resilience_part = "" if self.resilience == (1, 1) else f", resilience={resilience_str}"

        return f"Orderbook({self.name!r}, depth={depth_str}, width={width_str}, spread={spread_str}{k_part}{resilience_part})"
