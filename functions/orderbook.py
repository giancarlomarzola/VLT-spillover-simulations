class Orderbook:
    """
    Represents a limit orderbook with asymmetry between bid and ask sides.

    Parameters:
        name: Human-readable name for this orderbook.
        depth_bid: Market depth on bid side in tokens.
        depth_ask: Market depth on ask side in tokens.
        width_bid: Maximum slippage at full depth on bid side (in decimal, e.g., 0.01 for 1%).
        width_ask: Maximum slippage at full depth on ask side (in decimal).
        spread_bid: Spread at zero depth on bid side (in decimal, e.g., 0.005 for 0.5%).
        spread_ask: Spread at zero depth on ask side (in decimal).
        k_bid: Curvature parameter for slippage formula on bid side. k=1 gives linear slippage; lower k gives more convex curve.
        k_ask: Curvature parameter for slippage formula on ask side.
        resilience_bid: Resilience parameter on bid side (float between 0 and 1). 1=perfect resilience.
        resilience_ask: Resilience parameter on ask side.
    """

    def __init__(
        self,
        name,
        depth_bid,
        depth_ask,
        width_bid,
        width_ask,
        spread_bid,
        spread_ask,
        k_bid=1,
        k_ask=1,
        resilience_bid=1,
        resilience_ask=1
        ):
        if not isinstance(name, str):
            raise TypeError(f"name must be string, got {type(name)}")

        self.name = name
        self.depth_bid = depth_bid
        self.depth_ask = depth_ask
        self.width_bid = width_bid
        self.width_ask = width_ask
        self.spread_bid = spread_bid
        self.spread_ask = spread_ask
        self.k_bid = k_bid
        self.k_ask = k_ask
        self.resilience_bid = resilience_bid
        self.resilience_ask = resilience_ask

    def calculate_slippage(self, d, effective_price):
        """
        Calculate signed slippage s_t for a trade of size d (in tokens) at the given
        effective price, by inverting the synthetic orderbook f(z) = D * t/(k + (1-k)*t).
        Automatically selects bid side (d < 0) or ask side (d > 0) based on trade direction.
        """
        if d == 0:
            return 0.0

        if d > 0:
            D = self.depth_ask / effective_price
            S = self.spread_ask
            W = self.width_ask
            K = self.k_ask
        else:
            D = self.depth_bid / effective_price
            S = self.spread_bid
            W = self.width_bid
            K = self.k_bid

        x = abs(d)
        if D == 0 or x > D:
            s = W
        else:
            s = S + (W - S) * (K * x) / (D - (1 - K) * x)
        return s if d > 0 else -s

    def __repr__(self):
        k_part = "" if (self.k_bid == 1 and self.k_ask == 1) else f", k_bid={self.k_bid}, k_ask={self.k_ask}"
        resilience_part = "" if (self.resilience_bid == 1 and self.resilience_ask == 1) else f", resilience_bid={self.resilience_bid}, resilience_ask={self.resilience_ask}"

        return (f"Orderbook({self.name!r}, depth_bid={self.depth_bid}, depth_ask={self.depth_ask}, "
                f"width_bid={self.width_bid * 100}, width_ask={self.width_ask * 100}, "
                f"spread_bid={self.spread_bid * 100}, spread_ask={self.spread_ask * 100}"
                f"{k_part}{resilience_part})")
