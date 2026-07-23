import pandas as pd
import numpy as np


# (internal_key, output_column_suffix) — order matches the original schema
_TOKEN_VARS = [
    ("v", "v"),
    ("x", "x"),
    ("x_star", "xst"),
    ("lam", "lambda"),
    ("lam_star", "lambdast"),
    ("target_delta", "target_delta"),
    ("actual_delta", "actual_delta"),
]
_SIDES = (("up", 1), ("down", -1))


def _columns():
    cols = []
    for side, _ in _SIDES:
        cols += [f"{suffix}_{side}" for _, suffix in _TOKEN_VARS]
    cols += ["target_total_delta", "actual_total_delta", "price_multiplier", "orderbook_effect"]
    return cols


COLUMNS = _columns()
COL = {name: i for i, name in enumerate(COLUMNS)}


def _as_side_pair(param):
    """Scalar -> symmetric (bid, ask); 2-tuple -> asymmetric (bid, ask) as given."""
    if np.isscalar(param):
        return (param, param)
    bid, ask = param
    return (bid, ask)


def _slippage(d, depth, spread, width):
    """
    Signed slippage s_t for a trade of size d (in tokens), by inverting the
    piecewise-linear synthetic orderbook f(z). depth/spread/width are each
    (bid, ask) pairs, so bid- and ask-side books can differ (asymmetric case).

    Note: the closed-form inversion in the source material is only spelled
    out for the symmetric book. This applies it per-side (ask-side D,S,W for
    d>0, bid-side for d<0), which is the natural generalisation but is my
    extrapolation, not something stated explicitly — flag if that's not
    what you intended.
    """
    if d == 0:
        return 0.0
    D, S, W = (depth[1], spread[1], width[1]) if d > 0 else (depth[0], spread[0], width[0])
    x = abs(d)
    if D == 0:
        s = W
    else:
        s = W if x > D else x * (W - S) / D + S
    return s if d > 0 else -s


def _step_token(prev, omega, ret, lambda_target, lambda_up, lambda_down):
    """Advance one token (UP or DOWN) by one time step."""
    v = prev["v"] * (1 + omega * prev["lam_star"] * ret)
    x = prev["x_star"] * (1 + ret)

    if v <= 0:
        # Wipeout: liquidate remaining position (eqn 14 applies here too)
        liquidation_delta = -x
        return {"v": 0.0, "x": 0.0, "x_star": 0.0, "lam": 0.0, "lam_star": 0.0,
                "target_delta": liquidation_delta, "actual_delta": liquidation_delta}

    lam = omega * x / v

    if lambda_down <= lam <= lambda_up:
        x_star = x
    elif lambda_target is None:
        target_lambda = lambda_down if lam <= lambda_down else lambda_up
        x_star = omega * v * target_lambda
    else:
        x_star = omega * v * lambda_target

    lam_star = omega * x_star / v
    target_delta = x_star - x  # target rebalance size, in $

    # actual_delta will be determined by execution cap in run_simulation
    return {"v": v, "x": x, "x_star": x_star, "lam": lam, "lam_star": lam_star,
            "target_delta": target_delta, "actual_delta": target_delta}


def run_simulation(
    lambda_target, # None = boundary rebalancing, float = target rebalancing
    lambda_up, # upper boundary
    lambda_down, # lower boundary
    orderbook=None,
    data=None,
    currency=None,
    frequency=None,
    ):
    """
    Simulate a pair of variable-leverage UP/DOWN tokens through time.

    lambda_target: None uses boundary rebalancing (snap to upper/lower bound),
    any float uses target rebalancing (rebalance to that target leverage).
    lambda_up: upper boundary
    lambda_down: lower boundary

    orderbook: dict with keys "depth", "spread", "width", each a scalar
    (symmetric book) or a (bid, ask) pair (asymmetric book), in the units
    of the source equations (depth in tokens). Leave as None to run with
    no price impact. Slippage and the price multiplier follow the
    one-period execution lag of eqn 5: a rebalance at t only moves the
    price used at t+1.

    data: optional pre-loaded DataFrame. If provided, uses this directly.
    Otherwise requires currency and frequency to load pre-processed data.

    currency: ticker (e.g., "btc"). Required if data is None.
    frequency: sampling frequency (e.g., "30s", "1min"). Required if data is None.
    """

    if data is None:
        if currency is None or frequency is None:
            raise ValueError("Must provide either 'data' or both 'currency' and 'frequency'")

        filepath = f"dissertation_data/token_dataframes/{currency}_{frequency}_processed.parquet"
        try:
            df = pd.read_parquet(filepath)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Pre-processed data not found: {filepath}\n"
                f"Run data_processing.py to prepare {currency} data at {frequency} frequency."
            )
    else:
        df = data

    # Get initial
    price = np.asarray(df["price"].values, dtype=float)

    n = len(price)
    out = np.zeros((n, len(COLUMNS)))

    has_orderbook = orderbook is not None
    if has_orderbook:
        depth = _as_side_pair(orderbook["depth"])
        spread_pair = _as_side_pair(orderbook["spread"])
        width_pair = _as_side_pair(orderbook["width"])
        spread = (spread_pair[0] / 100, spread_pair[1] / 100)
        width = (width_pair[0] / 100, width_pair[1] / 100)

    # t = 0: Seed from basket primitives
    for side, omega in _SIDES:
        if side == "up":
            v0 = df["nTokensUP"].iloc[0] * df["up_price"].iloc[0]   # investment in basket currency
            x_star_0 = df["BasketUP"].iloc[0] * df["price"].iloc[0]  # already signed
        else:
            v0 = df["nTokensDOWN"].iloc[0] * df["down_price"].iloc[0]
            x_star_0 = df["BasketDOWN"].iloc[0] * df["price"].iloc[0]  # already signed

        # Compute initial leverage from basket
        lam_0_star = omega * x_star_0 / v0 if v0 > 0 else 0.0

        # Apply initial leverage bounds check (eqn 13): if lam_0* outside [lambda_down, lambda_up], reset to bound
        if lam_0_star < lambda_down:
            lam_0_star = lambda_down
        elif lam_0_star > lambda_up:
            lam_0_star = lambda_up

        out[0, COL[f"v_{side}"]] = v0
        out[0, COL[f"x_{side}"]] = x_star_0
        out[0, COL[f"xst_{side}"]] = x_star_0
        out[0, COL[f"lambda_{side}"]] = lam_0_star
        out[0, COL[f"lambdast_{side}"]] = lam_0_star
    out[0, COL["price_multiplier"]] = 1.0

    m_lag = 1.0  # m_{-1}, needed for the eqn 5 lag at t = 1

    for t in range(1, n):
        m_prev = out[t - 1, COL["price_multiplier"]]  # m_{t-1}

        # Guard against division by zero from zero/corrupted prices or multipliers
        if price[t - 1] == 0 or price[t] == 0 or m_lag == 0 or m_prev == 0:
            # Can't compute return; mark as zero for safety (skip rebalancing this period)
            ret = 0.0
        else:
            if has_orderbook:
                ret = (price[t] * m_prev) / (price[t - 1] * m_lag) - 1
            else:
                ret = price[t] / price[t - 1] - 1

        target_total = 0.0
        step_results = {}
        for side, omega in _SIDES:
            prev = {key: out[t - 1, COL[f"{suffix}_{side}"]] for key, suffix in _TOKEN_VARS}
            res = _step_token(prev, omega, ret, lambda_target, lambda_up, lambda_down)
            step_results[side] = res
            target_total += res["target_delta"]

        # Execution cap: if net trade exceeds available depth on either side, scale both sides (eqn 14)
        if has_orderbook and target_total != 0 and price[t] * m_prev != 0:
            d_target = target_total / (price[t] * m_prev)  # trade size, in tokens
            # Convert USD depth to tokens (the conversion and reconversion cancel exactly)
            depth_tokens = (depth[0] / (price[t] * m_prev), depth[1] / (price[t] * m_prev))
            D_side = depth_tokens[1] if d_target > 0 else depth_tokens[0]

            if abs(d_target) > D_side:
                # Execution capped: scale both sides proportionally, price moves by full width (eqn 14, sign-safe)
                scale = D_side / abs(d_target)
                s_t = np.sign(d_target) * width[1 if d_target > 0 else 0]  # full width on impact side
            else:
                # Execution uncapped: normal slippage
                scale = 1.0
                s_t = _slippage(d_target, depth_tokens, spread, width)
        else:
            scale = 1.0
            s_t = 0.0

        # Apply execution cap and recompute x_star, lam_star post-execution
        # The realised x_star can legitimately end up outside [lambda_down, lambda_up] bounds
        actual_total_delta = 0.0
        for side, omega in _SIDES:
            res = step_results[side]
            res["actual_delta"] = res["target_delta"] * scale
            res["x_star"] = res["x"] + res["actual_delta"]
            res["lam_star"] = omega * res["x_star"] / res["v"] if res["v"] > 0 else 0.0
            actual_total_delta += res["actual_delta"]

        for side, _ in _SIDES:
            res = step_results[side]
            for key, suffix in _TOKEN_VARS:
                out[t, COL[f"{suffix}_{side}"]] = res[key]

        out[t, COL["target_total_delta"]] = target_total
        out[t, COL["actual_total_delta"]] = actual_total_delta
        m_new = (1 + s_t) * m_prev
        # Clamp to prevent multiplier from going non-positive (would cause NaN/inf in next iteration)
        out[t, COL["price_multiplier"]] = max(m_new, 1e-10)
        out[t, COL["orderbook_effect"]] = s_t

        m_lag = m_prev  # becomes m_{t-1}, needed as the lag term at t+1

    return pd.DataFrame(out, columns=COLUMNS)


