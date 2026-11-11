class VLETP:
    """
    A single variable leverage exchange traded product (VLETP), e.g. one UP or DOWN
    variable leverage token (VLT).

    The product holds a notional exposure to an underlying and lets its leverage drift
    freely inside a band [leverage_lower_bound, leverage_upper_bound]. Only when leverage leaves the band
    does it rebalance, either to the nearest boundary or to a fixed target leverage.

    Parameters:
        name: Human-readable name for this product (e.g. "BTCUP").
        start_nav: Initial net asset value, in quote currency (v_0). Must be positive.
        start_leverage: Initial leverage, signed like the band (positive for a long
            product, negative for a short product). Initial notional exposure (x*_0)
            is start_leverage * start_nav. May lie outside the leverage band, in which
            case the product rebalances on its first step like any other breach.
        leverage_lower_bound: Lower boundary of the leverage band.
        leverage_upper_bound: Upper boundary of the leverage band.
            The sign of the band sets the direction of the product: both boundaries
            positive for a long (UP) product, e.g. [1.25, 4], both negative for a short
            (DOWN) product, e.g. [-4, -1.25]. They must share the same sign.
        leverage_target: Leverage to rebalance to once the band is breached, signed like
            the band. None means boundary rebalancing (snap back to whichever boundary
            was breached). A single number always rebalances to that leverage. A tuple
            (lower target, upper target) rebalances to the nearest of the two: the lower
            target after a lower-boundary breach, the upper target after an upper-boundary
            breach, e.g. band [1, 3] with targets (1.5, 2.5).

    Derived from the band:
        direction: +1 for a long product, -1 for a short product (omega).
        leverage_lower_bound, leverage_upper_bound, leverage_target: stored as magnitudes, so leverage_lower_bound
            is always the boundary closest to zero, whichever way round a short band
            was passed in.

    State (updated in place by the methods below):
        nav: Current NAV (v).
        exposure: Signed notional before the current step's rebalance (x).
        exposure_after: Signed notional after the current step's rebalance (x*).
        leverage: Leverage before the rebalance (lambda), always quoted as a positive number.
        leverage_after: Leverage after the rebalance (lambda*).
        target_delta: Rebalance trade the product wants, in quote currency (signed; >0 buys).
        actual_delta: Rebalance trade actually executed (smaller if capped by the orderbook).

    One time step is: apply_return() -> calculate_rebalance() -> execute_rebalance().
    Execution is a separate call because the execution cap depends on the net trade of
    every product hitting the same orderbook, which a single product cannot know.
    """

    def __init__(
        self,
        name,
        start_nav,
        start_leverage,
        leverage_lower_bound,
        leverage_upper_bound,
        leverage_target=None, # None: nearest boundary; number: fixed target; (lower, upper): nearest of the two
    ):
        if not isinstance(name, str):
            raise TypeError(f"name must be string, got {type(name)}")
        if start_nav <= 0:
            raise ValueError(f"start_nav must be positive, got {start_nav}")

        # Direction (omega) comes from the sign of the leverage band
        band = f"[{leverage_lower_bound}, {leverage_upper_bound}]"
        if leverage_lower_bound == 0 or leverage_upper_bound == 0:
            raise ValueError(f"leverage bounds must be non-zero, got {band}")
        if (leverage_lower_bound > 0) != (leverage_upper_bound > 0):
            raise ValueError(f"leverage_lower_bound and leverage_upper_bound must have the same sign, got {band}")
        direction = 1 if leverage_lower_bound > 0 else -1

        if direction == 1 and leverage_lower_bound > leverage_upper_bound:
            raise ValueError(f"leverage_lower_bound must not exceed leverage_upper_bound, got {band}")
        band_min = min(leverage_lower_bound, leverage_upper_bound)
        band_max = max(leverage_lower_bound, leverage_upper_bound)
        if isinstance(leverage_target, (tuple, list)):
            if len(leverage_target) != 2:
                raise ValueError(
                    "leverage_target must be None, a number or a (lower target, upper target) pair, "
                    f"got {leverage_target}"
                )
            targets = tuple(leverage_target)
        elif leverage_target is None:
            targets = ()
        else:
            targets = (leverage_target,)
        for target in targets:
            if not band_min <= target <= band_max:
                raise ValueError(f"leverage_target ({target}) must lie inside the leverage band {band}")
        if direction * start_leverage < 0:
            raise ValueError(
                f"start_leverage ({start_leverage}) must have the same sign as the leverage band {band}"
            )

        self.name = name
        self.direction = direction
        self.start_nav = float(start_nav)
        self.start_leverage = float(start_leverage)
        # Stored as magnitudes; a short band may be given as [-4, -1.25] or [-1.25, -4]
        self.leverage_lower_bound = min(abs(leverage_lower_bound), abs(leverage_upper_bound))
        self.leverage_upper_bound = max(abs(leverage_lower_bound), abs(leverage_upper_bound))
        # None, a single magnitude, or a (lower target, upper target) pair of magnitudes
        if len(targets) == 2:
            self.leverage_target = tuple(sorted(abs(target) for target in targets))
        else:
            self.leverage_target = abs(targets[0]) if targets else None

        self.reset()

    @classmethod
    def from_basket(
        cls,
        name,
        n_tokens,
        token_price,
        basket,
        underlying_price,
        leverage_lower_bound,
        leverage_upper_bound,
        leverage_target=None,
    ):
        """
        Build a product from issuer-style holdings data instead of NAV and leverage.

        n_tokens: Tokens outstanding.
        token_price: Price of one token, in quote currency.
        basket: Signed units of the underlying held per the issuer's basket (negative for short).
        underlying_price: Price of the underlying, in quote currency.
        """
        return cls(
            name=name,
            start_nav=n_tokens * token_price,
            start_leverage=(basket * underlying_price) / (n_tokens * token_price),
            leverage_lower_bound=leverage_lower_bound,
            leverage_upper_bound=leverage_upper_bound,
            leverage_target=leverage_target,
        )

    def reset(self):
        """Restore the product to its initial state."""
        self.nav = self.start_nav
        self.exposure = self.start_leverage * self.start_nav
        self.exposure_after = self.exposure

        self.leverage = abs(self.start_leverage)
        self.leverage_after = self.leverage

        self.target_delta = 0.0
        self.actual_delta = 0.0

    @property
    def is_wiped_out(self):
        """True once NAV has hit zero. A wiped-out product stays dead."""
        return self.nav <= 0

    @property
    def rebalancing_mode(self):
        """'boundary' (snap to the breached boundary) or 'target' (rebalance to leverage_target)."""
        return "boundary" if self.leverage_target is None else "target"

    def apply_return(self, ret):
        """
        Mark the product to market for an underlying return of `ret` (decimal, e.g. 0.01),
        updating NAV, pre-rebalance exposure and pre-rebalance leverage.

        If the move wipes out the NAV, the remaining position is liquidated: the state is
        zeroed and target_delta is set to the liquidation trade.
        """
        nav = self.nav * (1 + self.direction * self.leverage_after * ret)
        exposure = self.exposure_after * (1 + ret)

        if nav <= 0:
            self.nav = 0.0
            self.exposure = 0.0
            self.exposure_after = 0.0
            self.leverage = 0.0
            self.leverage_after = 0.0
            self.target_delta = -exposure  # liquidate whatever is left
            self.actual_delta = self.target_delta
            return

        self.nav = nav
        self.exposure = exposure
        self.exposure_after = exposure
        self.leverage = self.direction * exposure / nav
        self.leverage_after = self.leverage
        self.target_delta = 0.0
        self.actual_delta = 0.0

    def needs_rebalance(self):
        """True if current leverage (as a magnitude) sits outside [leverage_lower_bound, leverage_upper_bound]."""
        if self.is_wiped_out:
            return False
        return not self.leverage_lower_bound <= self.leverage <= self.leverage_upper_bound

    def rebalance_leverage(self):
        """
        Leverage the product wants to hold after this step: unchanged while inside the
        band, otherwise the breached boundary (boundary mode), the single leverage_target, or
        the nearest of the (lower target, upper target) pair (target mode).
        """
        if not self.needs_rebalance():
            return self.leverage
        if self.leverage_target is None:
            candidates = (self.leverage_lower_bound, self.leverage_upper_bound)
        elif isinstance(self.leverage_target, tuple):
            candidates = self.leverage_target
        else:
            return self.leverage_target
        # Nearest candidate: the lower one after a lower breach, the upper one after an upper breach
        return min(candidates, key=lambda candidate: abs(candidate - self.leverage))

    def calculate_rebalance(self):
        """
        Work out the rebalance trade needed after apply_return(), in quote currency
        (signed: >0 buys the underlying, <0 sells). Stores it as target_delta and returns it.
        Zero while leverage is inside the band.
        """
        if self.is_wiped_out:
            return self.target_delta  # liquidation trade set by apply_return (0 once dead)

        if self.needs_rebalance():
            target_exposure = self.direction * self.nav * self.rebalance_leverage()
            self.target_delta = target_exposure - self.exposure
        else:
            self.target_delta = 0.0
        self.actual_delta = self.target_delta
        return self.target_delta

    def execute_rebalance(self, scale=1.0):
        """
        Execute the rebalance computed by calculate_rebalance().

        scale: Fraction of target_delta that actually fills (1 = full execution). Below 1
            when the orderbook cannot absorb the trade, in which case post-rebalance
            leverage can legitimately end up outside the band.

        Returns the executed trade (actual_delta).
        """
        self.actual_delta = self.target_delta * scale
        if not self.is_wiped_out:
            self.exposure_after = self.exposure + self.actual_delta
            self.leverage_after = self.direction * self.exposure_after / self.nav
        return self.actual_delta

    def step(self, ret, scale=1.0):
        """
        Convenience for a product trading alone: mark to market, compute the rebalance
        and execute it in one go. Returns the executed trade.
        """
        self.apply_return(ret)
        self.calculate_rebalance()
        return self.execute_rebalance(scale)

    def state(self):
        """Snapshot of the current state, keyed like the simulation's per-token variables."""
        return {
            "v": self.nav,
            "x": self.exposure,
            "x_star": self.exposure_after,
            "lam": self.leverage,
            "lam_star": self.leverage_after,
            "target_delta": self.target_delta,
            "actual_delta": self.actual_delta,
        }

    def __repr__(self):
        target_part = "" if self.leverage_target is None else f", leverage_target={self.leverage_target}"
        return (
            f"VLETP({self.name!r}, direction={self.direction}, nav={self.nav}, "
            f"exposure={self.exposure_after}, leverage={self.leverage_after}, "
            f"band=[{self.leverage_lower_bound}, {self.leverage_upper_bound}]{target_part})"
        )
