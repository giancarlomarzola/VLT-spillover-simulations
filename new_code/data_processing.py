import pandas as pd
import os


def get_column_names(dataname):
    # Returns a list with the column names corresponding to the chosen dataname
    # Used within 'load_raw_data' function
    if dataname=="aggTrade":
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
          "data.M": "Ignore"
        }  

    elif dataname=="bookTicker":
        Columns = {  
          "data.u": "updateId",  
          "data.s": "symbol",  
          "data.b": "best_bid",
          "data.B": "best_bid_qty",
          "data.a": "best_ask",  
          "data.E": "Event_time",
          "data.T": "Trade_time",
          "data.A": "best_ask_qty" 
        }  

    elif dataname=="trade":
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
          "data.M": "Ignore"
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
        df = (pd
              .read_csv(filename, compression='gzip', index_col=False)
              .drop(columns=['Unnamed: 0'])
              .rename(columns=colnames)
              )
        
        # Convert timestamps to datetime format
        for col in ["Event_time", "Trade_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit = "ms")
            
    
        return df
    
    else:
        raise Exception(f"{filename} doesn't exist in location")

def resample_data(df, token, resample_freq):
    df2 = (df
        .assign(timestamp = lambda df: df.Event_time)
        .assign(latency = lambda df: ((df.Event_time-df.Trade_time).dt.total_seconds()).astype(int)/1000)
        .rename(columns={'Price':'price'})
        .loc[:, ["timestamp", "price", "latency"]]
        .sort_values("timestamp")
        .set_index("timestamp")
        .resample(resample_freq, closed="right", label="right")
        .last()
        .reset_index(drop=False)
        .rename(columns = lambda x: f"{token}_{x}" if x != "timestamp" else x)
    )
    return df2

def reformat_data(df, token):
    df2 = (df
        .assign(timestamp = lambda df: df.Event_time)
        .assign(latency = lambda df: ((df.Event_time-df.Trade_time).dt.total_seconds()).astype(int)/1000)
        .rename(columns={'Price':'price'})
        .loc[:, ["timestamp", "price", "latency"]]
        .sort_values("timestamp")
        .set_index("timestamp")
        #.last()
        .reset_index(drop=False)
        .rename(columns = lambda x: f"{token}_{x}" if x != "timestamp" else x)
    )
    return df2

def process_data(df, token_name, resample_freq=None):
    df = (df
        .assign(
            timestamp=lambda d: d.Event_time,
            latency=lambda d: ((d.Event_time - d.Trade_time).dt.total_seconds()).astype(int) / 1000,
        )
        .rename(columns={"Price": "price"})
        .loc[:, ["timestamp", "price", "latency"]]
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    if resample_freq is not None:
        df = df.resample(resample_freq, closed="right", label="right").last()

    return (df
        .reset_index(drop=False)
        .rename(columns=lambda x: f"{token_name}_{x}" if x != "timestamp" else x)
    )

def create_currency_df(currency, resample_freq=None):
    # resample options: e.g. "30s", "min", "5min"
    token_up = load_raw_data(
        "dissertation_data", 
        "binance", 
        f"{currency}upusdt", 
        "2021-05-19", 
        "trade"
        )
    token_down = load_raw_data(
        "dissertation_data", 
        "binance", 
        f"{currency}downusdt", 
        "2021-05-19", 
        "trade"
        )
    token = load_raw_data(
        "dissertation_data", 
        "binance-futures", 
        f"{currency}usdt", 
        "2021-05-19", 
        "trade"
        )

    if False:
        if resample_freq:
            # Reduce data to minute instead of ms
            token_up = resample_data(token_up, f"{currency}_up", resample_freq)
            token_down = resample_data(token_down, f"{currency}_down", resample_freq)
            token = resample_data(token, currency, resample_freq)
        elif resample_freq is None:
            # put data in correct format, without resampling
            token_up = reformat_data(token_up, 'sushiup')
            token_down = reformat_data(token_down, 'sushidown')
            token = reformat_data(token, 'sushi')

    # Clean and resample data
    token_up = process_data(token_up, f"{currency}_up", resample_freq)
    token_down = process_data(token_down, f"{currency}_down", resample_freq)
    token = process_data(token, currency, resample_freq)


    # Merge all dataframes on timestamp using nearest-match merge
    token_merged = token_up
    token_merged = pd.merge_asof(token_merged, token_down, on="timestamp")
    token_merged = pd.merge_asof(token_merged, token, on="timestamp")

    return token_merged


if __name__ == "__main__":
    # Example usage: Load and process sushi token data
    print("Loading sushi token data without resampling...")
    sushi_data = create_currency_df("sushi", resample_freq=None)
    print(f"Data shape: {sushi_data.shape}")
    print(f"Columns: {list(sushi_data.columns)}")
    print(f"\nFirst few rows:\n{sushi_data.head()}")

    print("\n" + "="*50)
    print("Loading sushi token data with 1-minute resampling...")
    sushi_data_1m = create_currency_df("sushi", resample_freq="1T")
    print(f"Data shape: {sushi_data_1m.shape}")
    print(f"Columns: {list(sushi_data_1m.columns)}")
    print(f"\nFirst few rows:\n{sushi_data_1m.head()}")

