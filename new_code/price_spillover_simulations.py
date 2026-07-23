import pandas as pd
import numpy as np

currency = "btc"
frequency = "30s"

# Parameters (same for up and down)
lambda_target = None  # None = boundary rebalancing, float = target rebalancing
lambda_up = 4  # upper boundary
lambda_down = 1.25  # lower boundary
# lambda_0     = 3.5 # Leverage at start


def import_data(currency, frequency):
    # import dataframe
    filename = f"{currency}_data" + (f"_{frequency}" if frequency else "")
    df = pd.read_csv(f"dissertation_data/token_dataframes/{filename}.csv")


    # Ensure timestamp is datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filter to analysis period
    analysis_start = pd.Timestamp("2021-05-19 12:00:00")
    analysis_end = pd.Timestamp("2021-05-19 14:00:00")

    df = df[df["timestamp"].between(analysis_start, analysis_end)].reset_index(drop=True)
    return df


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
    s = W if x > D else x * (W - S) / D + S
    return s if d > 0 else -s


def _step_token(prev, omega, ret, lambda_target, lambda_up, lambda_down):
    """Advance one token (UP or DOWN) by one time step."""
    v = prev["v"] * (1 + omega * prev["lam_star"] * ret)
    if v <= 0:
        return dict.fromkeys(("v", "x", "x_star", "lam", "lam_star", "target_delta", "actual_delta"), 0.0)

    x = prev["x_star"] * (1 + ret)
    lam = omega * x / v

    if lambda_down <= lam <= lambda_up:
        x_star = x
    elif lambda_target is None:
        target_lambda = lambda_down if lam <= lambda_down else lambda_up
        x_star = omega * v * target_lambda
    else:
        x_star = omega * v * lambda_target

    lam_star = omega * x_star / v
    target_delta = x_star - x  # in $

    return {"v": v, "x": x, "x_star": x_star, "lam": lam, "lam_star": lam_star,
            "target_delta": target_delta, "actual_delta": target_delta}


def run_simulation(
    df, # cleaned binance data df
    lambda_target, # None = boundary rebalancing, float = target rebalancing
    lambda_up, # upper boundary
    lambda_down, # lower boundary
    orderbook=None,
    ):
    """
    Simulate a pair of variable-leverage UP/DOWN tokens through time.

    lambda_target: None uses boundary rebalancing (snap to upper/lower bound),
    any float uses target rebalancing (rebalance to that target leverage).

    orderbook: dict with keys "depth", "spread", "width", each a scalar
    (symmetric book) or a (bid, ask) pair (asymmetric book), in the units
    of the source equations (depth in tokens). Leave as None to run with
    no price impact. Slippage and the price multiplier follow the
    one-period execution lag of eqn 5: a rebalance at t only moves the
    price used at t+1.
    """

    # Get initial 
    price = np.asarray(df["price"].values, dtype=float)

    # UP parameters
    up_investment = df["nTokensUP"].iloc[0] * df["up_price"].iloc[0]
    up_lambda_0 = df["leverageUP"].iloc[0]
    # DOWN Parameters
    down_investment = df["nTokensDOWN"].iloc[0] * df["down_price"].iloc[0]
    down_lambda_0 = df["leverageDOWN"].iloc[0]

    n = len(price)
    out = np.zeros((n, len(COLUMNS)))

    has_orderbook = orderbook is not None
    if has_orderbook:
        depth = _as_side_pair(orderbook["depth"])
        spread_pair = _as_side_pair(orderbook["spread"])
        width_pair = _as_side_pair(orderbook["width"])
        spread = (spread_pair[0] / 100, spread_pair[1] / 100)
        width = (width_pair[0] / 100, width_pair[1] / 100)

    # t = 0
    for side, omega in _SIDES:
        investment = up_investment if side == "up" else down_investment
        lam_0 = up_lambda_0 if side == "up" else down_lambda_0
        x0 = omega * investment * lam_0
        out[0, COL[f"v_{side}"]] = investment
        out[0, COL[f"x_{side}"]] = x0
        out[0, COL[f"xst_{side}"]] = x0
        out[0, COL[f"lambda_{side}"]] = omega * x0 / investment
        out[0, COL[f"lambdast_{side}"]] = omega * x0 / investment
    out[0, COL["price_multiplier"]] = 1.0

    m_lag = 1.0  # m_{-1}, needed for the eqn 5 lag at t = 1

    for t in range(1, n):
        m_prev = out[t - 1, COL["price_multiplier"]]  # m_{t-1}

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

        if has_orderbook and target_total != 0:
            d_t = target_total / (price[t] * m_prev)  # trade size, in tokens
            s_t = _slippage(d_t, depth, spread, width)
        else:
            s_t = 0.0

        for side, _ in _SIDES:
            res = step_results[side]
            for key, suffix in _TOKEN_VARS:
                out[t, COL[f"{suffix}_{side}"]] = res[key]

        out[t, COL["target_total_delta"]] = target_total
        out[t, COL["actual_total_delta"]] = target_total  # eqn 5 caps slippage, not execution size
        out[t, COL["price_multiplier"]] = (1 + s_t) * m_prev
        out[t, COL["orderbook_effect"]] = s_t

        m_lag = m_prev  # becomes m_{t-1}, needed as the lag term at t+1

    return pd.DataFrame(out, columns=COLUMNS)


if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"Simulations for {currency} {frequency}")
    print(f"{'='*60}")

    # Baseline simulation with no orderbook
    print(f"\n{'='*60}")
    print("Running baseline simulation with no orderbook")
    print(f"{'='*60}")

    binance_data = import_data(currency, frequency)


    result = run_simulation(
        binance_data,
        lambda_target=lambda_target,
        lambda_up=lambda_up,
        lambda_down=lambda_down,
        orderbook=None
    )

    output_path = f"dissertation_data/results/{currency}_{frequency}_simulation_no_orderbook.csv"
    result.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print(f"Shape: {result.shape}")

    # Parameter combinations for orderbooks
    D_vals = [1_000_000, 50_000_000]  # in USD
    W_vals = [1, 10]  # in percent
    S_vals = [0.0005, 0.005]  # in percent

    combinations = [
        (D, W, S)
        for D in D_vals
        for W in W_vals
        for S in S_vals
    ]

    # Run simulations for each orderbook combination
    for orderbook_idx, (D, W, S) in enumerate(combinations, 1):
        print(f"\n{'='*60}")
        print(f"Running simulation for orderbook {orderbook_idx}: D={D}, W={W}%, S={S}%")
        print(f"{'='*60}")

        # Configure orderbook with current parameters
        orderbook = {
            "depth": D,
            "spread": S,
            "width": W
        }

        result = run_simulation(
            binance_data,
            lambda_target=lambda_target,
            lambda_up=lambda_up,
            lambda_down=lambda_down,
            orderbook=orderbook
        )

        # Save with orderbook index in filename
        output_path = f"dissertation_data/results/{currency}_{frequency}_simulation_orderbook_{orderbook_idx}.csv"
        result.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        print(f"Shape: {result.shape}")
