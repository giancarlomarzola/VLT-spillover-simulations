import time

import pandas as pd
import requests

from utils.api_key import API_KEY
from utils.paths import DATA_RAW

COINDESK_DIR = DATA_RAW / "coindesk_api"
COINDESK_DIR.mkdir(parents=True, exist_ok=True)

# API_KEY = API_KEY  # from https://developers.coindesk.com
BASE_URL = "https://data-api.coindesk.com/futures/v2/historical/trades/hour"


def fetch_hour(market: str, instrument: str, hour_ts: int) -> list[dict]:
    """Fetch all trades for one hour. hour_ts must be the Unix timestamp
    at the start of the hour (UTC, on the hour)."""
    params = {
        "market": market,  # e.g. "binance"
        "instrument": instrument,  # e.g. "BTC-USDT-VANILLA-PERPETUAL"
        "hour_ts": hour_ts,
        "groups": "ID,MAPPING,TRADE",  # verify allowed group names on the docs page
        "api_key": API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "Data" not in data:
        err_msg = data.get("Message") or data.get("Err") or data
        raise RuntimeError(f"API error for hour_ts={hour_ts}: {err_msg}")
    return data["Data"]


def fetch_range(
    market: str, instrument: str, start_ts: int, end_ts: int, pause: float = 0.2
) -> pd.DataFrame:
    """Fetch every hour between start_ts and end_ts (inclusive), both
    Unix timestamps aligned to the hour."""
    all_trades = []
    ts = start_ts
    while ts <= end_ts:
        try:
            trades = fetch_hour(market, instrument, ts)
        except (requests.HTTPError, RuntimeError) as e:
            print(f"Stopping fetch_range: {e}")
            break
        all_trades.extend(trades)
        ts += 3600
        time.sleep(pause)  # basic rate-limit courtesy; tune to your API tier
    return pd.DataFrame(all_trades)


if __name__ == "__main__":
    start_dt = pd.Timestamp("2026-08-02 12:00:00", tz="UTC")
    end_dt = pd.Timestamp("2026-08-02 12:00:00", tz="UTC")
    start = int(start_dt.timestamp())
    end = int(end_dt.timestamp())
    df = fetch_range("binance", "BTC-USDT-VANILLA-PERPETUAL", start, end)
    filename = f"btc_usdt_trades_{start_dt:%Y-%m-%d_%H%M}_{end_dt:%Y-%m-%d_%H%M}.csv"
    df.to_csv(COINDESK_DIR / filename, index=False)
    print(df.head())
