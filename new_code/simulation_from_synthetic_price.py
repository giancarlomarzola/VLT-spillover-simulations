import random
import pandas as pd
from new_code.price_spillover_simulations import run_simulation

# Random price series - to be substituted later
def generate_price_series(start_price=100, steps=100, volatility=0.01):
    prices = [start_price]
    for _ in range(steps - 1):
        ret = random.uniform(-volatility, volatility)
        prices.append(round(prices[-1] * (1+ret), 2))
    return prices


price_series = generate_price_series(50_000, 144_000)
df = pd.DataFrame(price_series)

df.plot()


lambda_target = None  # None = boundary rebalancing, float = target rebalancing
lambda_upper = 4  # upper boundary
lambda_lower = 1.25  # lower boundary

start_nav_up = 30_000_000
start_exposure_up = 90_000_000

start_nav_down = 30_000_000
start_exposure_down = -90_000_000

# Orderbooks
# width and spread in %
orderbook_formula = "curved"
k = 0.5

orderbooks = {
    "Deep Narrow Tight"     : {"depth":50_000_000, "width":1,  "spread":0.005},
}

result = run_simulation(
    lambda_target=lambda_target,
    lambda_upper=lambda_upper,
    lambda_lower=lambda_lower,
    orderbook=orderbooks["Deep Narrow Tight"],
    orderbook_formula=orderbook_formula,
    k=k,
    price_series=price_series,
    start_nav_up=start_nav_up,
    start_exposure_up=start_exposure_up,
    start_nav_down=start_nav_down,
    start_exposure_down=start_exposure_down,
)
