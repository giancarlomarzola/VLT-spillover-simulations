import os

import pandas as pd

from utils.paths import DATA_PROCESSED, DATA_RAW

# Available data frequencies for simulations
FREQUENCIES = ["Tick", "50ms", "500ms", "1s", "15s", "30s", "1min"]


def get_column_names(dataname):
    # Returns a list with the column names corresponding to the chosen dataname
    # Used within 'load_raw_data' function
    if dataname == "aggTrade":
        Columns = {
            "data.e": "Event_type",
            "data.E": "Event_time",
            "data.s": "Symbol",
            "data.a": "Aggregate_trade_ID",
            "data.p": "Price",
            "data.q": "Quantity",
            "data.f": "First_trade_ID",
            "data.l": "Last_trade_ID",
            "data.T": "Trade_time",
            "data.m": "buyer_maker_flag",
            "data.M": "Ignore",
        }

    elif dataname == "bookTicker":
        Columns = {
            "data.u": "updateId",
            "data.s": "symbol",
            "data.b": "best_bid",
            "data.B": "best_bid_qty",
            "data.a": "best_ask",
            "data.E": "Event_time",
            "data.T": "Trade_time",
            "data.A": "best_ask_qty",
        }

    elif dataname == "trade":
        Columns = {
            "data.e": "Event_type",
            "data.E": "Event_time",
            "data.s": "Symbol",
            "data.t": "Trade_ID",
            "data.p": "Price",
            "data.q": "Quantity",
            "data.b": "Buyer_order_ID",
            "data.a": "Seller order ID",
            "data.T": "Trade_time",
            "data.m": "buyer_maker_flag",
            "data.M": "Ignore",
        }
    else:
        raise Exception(f"dataname {dataname} in function get_column_names not found")  # noqa: TRY002

    return Columns


def load_raw_data(filepath, tradevenue, ticker, timestamp, dataname):
    # Loads the specified data
    filename = f"{filepath}/{tradevenue}/{timestamp}/{dataname}/{tradevenue}_{ticker}_{dataname}_{timestamp}.csv.gz"

    if os.path.exists(filename):
        # Get column names from previous function
        colnames = get_column_names(dataname)
        # Read data, drop 'unnamed' column and rename remaining columns
        df = (
            pd.read_csv(filename, compression="gzip", index_col=False)
            .drop(columns=["Unnamed: 0"], errors="ignore")
            .rename(columns=colnames)
        )

        # Convert timestamps to datetime format
        for col in ["Event_time", "Trade_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit="ms")

        return df

    else:
        raise Exception(f"{filename} doesn't exist in location")  # noqa: TRY002


def process_data(df, token_name):
    """Clean trade data: extract timestamp and price, sort, and prefix with token_name."""
    df = (
        df.assign(
            timestamp=lambda d: d.Event_time,
            latency=lambda d: (d.Event_time - d.Trade_time).dt.total_seconds() * 1000,
        )
        .rename(columns={"Price": "price"})
        .loc[:, ["timestamp", "price", "latency"]]
        .sort_values("timestamp")
    )

    return df.rename(
        columns=lambda x: f"{token_name}{x}" if x != "timestamp" else x
    )


def add_rebalance_data(token_merged, currency):
    # Load rebalancing data from Excel
    excelfile = DATA_RAW / "Leverage Tokens" / "BinanceLeverageToken.xlsx"

    # Read rebalancing data for up and down tokens
    up_rebalance = pd.read_excel(
        excelfile, sheet_name=f"{currency.capitalize()}Up", usecols="A:H"
    )
    down_rebalance = pd.read_excel(
        excelfile, sheet_name=f"{currency.capitalize()}Down", usecols="A:H"
    )

    columns_to_clean = [
        "BasketBefore",
        "BasketAfter",
        "LeverageBefore",
        "LeverageAfter",
        "TokensBefore",
        "TokensAfter",
    ]

    # Clean data and convert to float
    for df in [up_rebalance, down_rebalance]:
        for col in columns_to_clean:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace("\xa0", "", regex=False)
                df[col] = df[col].str.replace("+", "", regex=False)
                df[col] = df[col].str.replace(",", "", regex=False).astype(float)

    # Sort by time
    up_rebalance = up_rebalance.sort_values("Time").reset_index(drop=True)
    down_rebalance = down_rebalance.sort_values("Time").reset_index(drop=True)

    # Convert Time and timestamp to matching dtype (ns) for merge_asof compatibility
    up_rebalance["Time"] = pd.to_datetime(up_rebalance["Time"]).dt.as_unit("ns")
    down_rebalance["Time"] = pd.to_datetime(down_rebalance["Time"]).dt.as_unit("ns")
    token_merged["timestamp"] = token_merged["timestamp"].dt.as_unit("ns")

    # Verify timezone: SUSHIUP should have rebalances in the known UTC window
    if currency.lower() == "sushi":
        known_window_start = pd.Timestamp("2021-05-19 13:53:55", tz="UTC")
        known_window_end = pd.Timestamp("2021-05-19 14:05:36", tz="UTC")
        times_utc = pd.to_datetime(up_rebalance["Time"], utc=True)
        in_window = times_utc.between(known_window_start, known_window_end).all()
        if not in_window:
            out_of_bounds = times_utc[~times_utc.between(known_window_start, known_window_end)]
            print("WARNING: Some SUSHIUP rebalance times are outside expected UTC window:")
            print(f"  Expected: {known_window_start} to {known_window_end}")
            print(f"  Found: {out_of_bounds.min()} to {out_of_bounds.max()}")

    # Merge rebalancing data to token_merged (basket and token counts only)
    token_merged = pd.merge_asof(
        token_merged,
        up_rebalance[["Time", "BasketAfter", "TokensAfter"]],
        left_on="timestamp",
        right_on="Time",
    ).rename(columns={"BasketAfter": "BasketUP", "TokensAfter": "nTokensUP"})
    token_merged = token_merged.drop(columns=["Time"])

    token_merged = pd.merge_asof(
        token_merged,
        down_rebalance[["Time", "BasketAfter", "TokensAfter"]],
        left_on="timestamp",
        right_on="Time",
    ).rename(columns={"BasketAfter": "BasketDOWN", "TokensAfter": "nTokensDOWN"})
    token_merged = token_merged.drop(columns=["Time"])

    # Verify BasketDOWN is negative (as expected)
    if (token_merged["BasketDOWN"] > 0).any():
        print("WARNING: BasketDOWN contains positive values; dropping the minus sign in leverage calculation")
        token_merged = token_merged.assign(
            leverageUP   = lambda d: (d.BasketUP   * d.price) / (d.nTokensUP   * d.up_price),
            leverageDOWN = lambda d: (d.BasketDOWN * d.price) / (d.nTokensDOWN * d.down_price),
        )
    else:
        # Derive mark-to-market leverage from basket (BasketDOWN is already negative)
        token_merged = token_merged.assign(
            leverageUP   = lambda d: (d.BasketUP   * d.price) / (d.nTokensUP   * d.up_price),
            leverageDOWN = lambda d: -(d.BasketDOWN * d.price) / (d.nTokensDOWN * d.down_price),
        )

    # Assert all required columns are present before returning
    required_cols = {"nTokensUP", "nTokensDOWN", "leverageUP", "leverageDOWN", "BasketUP", "BasketDOWN"}
    assert required_cols <= set(token_merged.columns), \
        f"Missing required columns: {required_cols - set(token_merged.columns)}"

    return token_merged


def create_currency_df(currency, include_rebalance=True):
    """Load and clean data for a leverage token pair, merging perpetual with UP/DOWN tokens."""
    token_up = load_raw_data(
        str(DATA_RAW), "binance", f"{currency}upusdt", "2021-05-19", "trade"
    )
    token_down = load_raw_data(
        str(DATA_RAW), "binance", f"{currency}downusdt", "2021-05-19", "trade"
    )
    token = load_raw_data(
        str(DATA_RAW), "binance-futures", f"{currency}usdt", "2021-05-19", "trade"
    )

    # Clean data at tick level (resampling deferred to simulation stage)
    token_up = process_data(token_up, "up_")
    token_down = process_data(token_down, "down_")
    token = process_data(token, "")

    # Merge all dataframes on timestamp; anchor on perpetual (exogenous driver)
    # Use merge_asof for tick-level data (best match on nearest timestamp)
    token_merged = pd.merge_asof(token, token_up, on="timestamp")
    token_merged = pd.merge_asof(token_merged, token_down, on="timestamp")

    # Add rebalancing data if requested
    if include_rebalance:
        token_merged = add_rebalance_data(token_merged, currency)

    return token_merged


def prepare_processed_data(currency, frequencies=None):
    """Load raw data, resample to specified frequencies, then filter to analysis period.

    Args:
        currency: ticker (e.g., "btc", "sushi")
        frequencies: list of frequencies (e.g., ["15s", "30s", "1min"]).
    """
    output_folder = DATA_PROCESSED
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Loading {currency.upper()} token data (tick-level, cleaned)...")
    data = create_currency_df(currency)
    print(f"Raw data shape: {data.shape}")

    # Ensure timestamp is datetime and UTC-aware
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)

    # Define analysis period (UTC)
    analysis_start = pd.Timestamp("2021-05-19 12:00:00", tz="UTC")
    analysis_end = pd.Timestamp("2021-05-19 14:00:00", tz="UTC")

    # Filter tick-level data to analysis period and save
    data_filtered = data[data["timestamp"].between(analysis_start, analysis_end)].reset_index(drop=True)
    print(f"Filtered data shape (2-hour window): {data_filtered.shape}")

    filename_tick = output_folder / f"{currency}_tick_processed.parquet"
    data_filtered.to_parquet(filename_tick, index=False)
    print(f"Saved tick-level data ({data_filtered.shape[0]} rows) to {filename_tick}")

    if frequencies is not None:
        # Resample full dataset first (for complete bins), then filter to analysis period
        for freq in frequencies:
            df_resampled = data.copy()
            df_resampled = df_resampled.set_index("timestamp")
            # Resample entire dataframe to create bins, take last value in each bin
            df_resampled = df_resampled.resample(freq, closed="right", label="right").last()
            # Forward-fill any gaps
            df_resampled = df_resampled.ffill()
            df_resampled = df_resampled.reset_index()

            # Filter resampled data to analysis period
            df_resampled = df_resampled[df_resampled["timestamp"].between(analysis_start, analysis_end)].reset_index(drop=True)

            filename = output_folder / f"{currency}_{freq}_processed.parquet"
            df_resampled.to_parquet(filename, index=False)
            print(f"Saved {freq} resampled data ({df_resampled.shape[0]} rows) to {filename}")


if __name__ == "__main__":
    currencies = ["sushi", "btc", "eth"]
    # Use all frequencies except Tick (Tick is loaded separately)
    frequencies = [f for f in FREQUENCIES if f != "Tick"]

    for currency in currencies:
        print(f"\n{'=' * 50}")
        print(f"Processing {currency.upper()}")
        print(f"{'=' * 50}")
        prepare_processed_data(currency, frequencies)
