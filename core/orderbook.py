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
        resilience_ask=1,
    ):
        if not isinstance(name, str):
            raise TypeError(f"name must be string, got {type(name)}")

        self.name = name
        self.depth_bid_full = depth_bid  # Original depth
        self.depth_ask_full = depth_ask
        self.depth_bid = depth_bid  # Current depth (can be depleted)
        self.depth_ask = depth_ask
        self.width_bid = width_bid
        self.width_ask = width_ask
        self.spread_bid = spread_bid
        self.spread_ask = spread_ask
        self.k_bid = k_bid
        self.k_ask = k_ask
        self.resilience_bid = resilience_bid
        self.resilience_ask = resilience_ask

    def _log_parameters(self):
        """Log orderbook parameters for verification."""
        print(f"\n{'=' * 60}")
        print(f"Orderbook: {self.name}")
        print(f"{'=' * 60}")
        print(f"  Depth (tokens):    bid={self.depth_bid_full:>15,.0f}  ask={self.depth_ask_full:>15,.0f}")
        print(f"  Width (decimal):   bid={self.width_bid:>15.6f}  ask={self.width_ask:>15.6f}")
        print(f"  Spread (decimal):  bid={self.spread_bid:>15.6f}  ask={self.spread_ask:>15.6f}")
        print(f"  Curvature (k):     bid={self.k_bid:>15.6f}  ask={self.k_ask:>15.6f}")
        print(f"  Resilience:        bid={self.resilience_bid:>15.6f}  ask={self.resilience_ask:>15.6f}")
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

    def replenish(self, time_delta):
        """
        Replenish orderbook depth on both sides between trades.
        Replenishes by resilience*depth_full per time step, capped at full depth.

        Args:
            time_delta: Elapsed time since the previous rebalancing trade (Δ)
        """
        self.depth_bid = min(
            self.depth_bid + self.resilience_bid * self.depth_bid_full * time_delta,
            self.depth_bid_full,
        )
        self.depth_ask = min(
            self.depth_ask + self.resilience_ask * self.depth_ask_full * time_delta,
            self.depth_ask_full,
        )

    def __repr__(self):
        k_part = "" if (self.k_bid == 1 and self.k_ask == 1) else f", k_bid={self.k_bid}, k_ask={self.k_ask}"
        resilience_part = (
            ""
            if (self.resilience_bid == 1 and self.resilience_ask == 1)
            else f", resilience_bid={self.resilience_bid}, resilience_ask={self.resilience_ask}"
        )

        return (
            f"Orderbook({self.name!r}, depth_bid={self.depth_bid}, depth_ask={self.depth_ask}, "
            f"width_bid={self.width_bid * 100}, width_ask={self.width_ask * 100}, "
            f"spread_bid={self.spread_bid * 100}, spread_ask={self.spread_ask * 100}"
            f"{k_part}{resilience_part})"
        )
