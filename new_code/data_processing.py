import pandas as pd
import os
from IPython.display import display


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
        Columns = {}
        # raise Exception(f"dataname {dataname} in function get_column_names not found")

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
            .drop(columns=["Unnamed: 0"])
            .rename(columns=colnames)
        )

        # Convert timestamps to datetime format
        for col in ["Event_time", "Trade_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit="ms")

        return df

    else:
        raise Exception(f"{filename} doesn't exist in location")


def process_data(df, token_name, resample_freq=None):
    df = (
        df.assign(
            timestamp=lambda d: d.Event_time,
            latency=lambda d: ((d.Event_time - d.Trade_time).dt.total_seconds()).astype(
                int
            )
            / 1000,
        )
        .rename(columns={"Price": "price"})
        .loc[:, ["timestamp", "price", "latency"]]
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    if resample_freq is not None:
        df = df.resample(resample_freq, closed="right", label="right").last()

    return df.reset_index(drop=False).rename(
        columns=lambda x: f"{token_name}_{x}" if x != "timestamp" else x
    )


def add_rebalance_data(token_merged, currency):
    # Load rebalancing data from Excel
    excelfile = "dissertation_data/Leverage Tokens/BinanceLeverageToken.xlsx"

    try:
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

        # Convert Time to same dtype as timestamp for merge compatibility
        up_rebalance["Time"] = pd.to_datetime(up_rebalance["Time"]).dt.as_unit("ms")
        down_rebalance["Time"] = pd.to_datetime(down_rebalance["Time"]).dt.as_unit("ms")

        # Merge rebalancing data to token_merged
        token_merged = pd.merge_asof(
            token_merged,
            up_rebalance[["Time", "BasketAfter", "TokensAfter"]],
            left_on="timestamp",
            right_on="Time",
        ).rename(columns={"BasketAfter": "BasketUP", "TokensAfter": "nTokensUP"})

        token_merged = pd.merge_asof(
            token_merged,
            down_rebalance[["Time", "BasketAfter", "TokensAfter"]],
            left_on="timestamp",
            right_on="Time",
        ).rename(columns={"BasketAfter": "BasketDOWN", "TokensAfter": "nTokensDOWN"})

        return token_merged

    except FileNotFoundError:
        print(
            f"Warning: Excel file {excelfile} not found. Returning token_merged without rebalance data."
        )
        return token_merged
    except Exception as e:
        print(
            f"Warning: Error loading rebalance data: {e}. Returning token_merged without rebalance data."
        )
        return token_merged


def create_currency_df(currency, resample_freq=None, include_rebalance=True):
    # resample options: e.g. "30s", "min", "5min"
    token_up = load_raw_data(
        "dissertation_data", "binance", f"{currency}upusdt", "2021-05-19", "trade"
    )
    token_down = load_raw_data(
        "dissertation_data", "binance", f"{currency}downusdt", "2021-05-19", "trade"
    )
    token = load_raw_data(
        "dissertation_data", "binance-futures", f"{currency}usdt", "2021-05-19", "trade"
    )

    # Clean and resample data
    token_up = process_data(token_up, f"{currency}_up", resample_freq)
    token_down = process_data(token_down, f"{currency}_down", resample_freq)
    token = process_data(token, currency, resample_freq)

    # Merge all dataframes on timestamp using nearest-match merge
    token_merged = token_up
    token_merged = pd.merge_asof(token_merged, token_down, on="timestamp")
    token_merged = pd.merge_asof(token_merged, token, on="timestamp")

    # Add rebalancing data if requested
    if include_rebalance:
        token_merged = add_rebalance_data(token_merged, currency)

    return token_merged


if __name__ == "__main__":

    # Create output folder if it doesn't exist
    output_folder = "dissertation_data/token_dataframes"
    os.makedirs(output_folder, exist_ok=True)

    for currency in ["sushi", "btc"]:
        # Load and process data without resampling
        print(f"Loading {currency.upper()} token data without resampling...")
        data = create_currency_df(currency)
        print(f"Data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        display(data.head())
        data.to_csv(f"{output_folder}/{currency}_data.csv", index=False)
        print(f"Saved to {output_folder}/{currency}_data.csv")

        print("\n" + "=" * 50)

        # Load and process data with 1-minute resampling
        print(f"Loading {currency.upper()} token data with 1-minute resampling...")
        data_1m = create_currency_df(currency, resample_freq="1min")
        print(f"Data shape: {data_1m.shape}")
        print(f"Columns: {list(data_1m.columns)}")
        display(data_1m.head())
        data_1m.to_csv(f"{output_folder}/{currency}_data_1m.csv", index=False)
        print(f"Saved to {output_folder}/{currency}_data_1m.csv")

        print("\n" + "=" * 50)
