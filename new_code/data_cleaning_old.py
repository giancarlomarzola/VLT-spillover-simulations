currency = 'sushi' # all lowercase
minute_data = True

filepath = "dissertation_data"
plot_path = "figures"

# Needed to save plots in correct location
model2_path, full = 'Model 2', '' # to save plots in correct location
if not minute_data:
    model2_path, full = 'Model 2 - Full Data', '_full'

from functools import reduce
import datetime


# Needed for plot
import matplotlib.dates as mdates
from matplotlib.dates import MinuteLocator # HourLocator, MonthLocator, YearLocator,


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
            
    
        return df, df.iloc[:100_000]
    
    else:
        raise Exception(f"{filename} doesn't exist in location")


def resample_data(df, token, time_interval='1T'):
    df2 = (df
             .assign(timestamp = lambda df: df.Event_time)
             .assign(latency = lambda df: ((df.Event_time-df.Trade_time).dt.total_seconds()).astype(int)/1000)
             .rename(columns={'Price':'price'})
             .loc[:, ["timestamp", "price", "latency"]]
             .sort_values("timestamp")
             .set_index("timestamp")
             .resample(time_interval, closed="right", label="right")
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


# --------------------------------------------------
#%% Import Token Data
#--------------------------------------------------

if minute_data: # Use minute data - set to False for all data
    # Load ETH data files
    sushiup, sushiup_     = load_raw_data(filepath, "binance", f"{currency}upusdt", "2021-05-19", "trade")
    sushidown, sushidown_ = load_raw_data(filepath, "binance", f"{currency}downusdt", "2021-05-19", "trade")
    sushi, sushi_         = load_raw_data(filepath, "binance-futures", f"{currency}usdt", "2021-05-19", "trade")
    
    # Reduce data to minute instead of ms
    sushiup_minute   = resample_data(sushiup, 'sushiup', '1T')
    sushidown_minute = resample_data(sushidown, 'sushidown', '1T')
    sushi_minute     = resample_data(sushi, 'sushi', '1T')
    
    # merge the minute data to one df (merge)
    sushi_merged = reduce(lambda left, right: pd.merge(left, right, on="timestamp"), 
        [sushiup_minute, sushidown_minute, sushi_minute])

else: # Use all data, rather than minute
        # Load ETH data files
    sushiup, sushiup_     = load_raw_data(filepath, "binance", f"{currency}upusdt", "2021-05-19", "trade")
    sushidown, sushidown_ = load_raw_data(filepath, "binance", f"{currency}downusdt", "2021-05-19", "trade")
    sushi, sushi_         = load_raw_data(filepath, "binance-futures", f"{currency}usdt", "2021-05-19", "trade")
    
    # put data in correct format, without resampling
    sushiup_minute   = reformat_data(sushiup, 'sushiup')
    sushidown_minute = reformat_data(sushidown, 'sushidown')
    sushi_minute     = reformat_data(sushi, 'sushi')
     
    # merge the data to one df (asof)
    sushi_merged = reduce(lambda left, right: pd.merge_asof(left, right, on="timestamp"), 
        [sushiup_minute, sushidown_minute, sushi_minute])



# --------------------------------------------------
#%% Import Rebalance Data
#--------------------------------------------------

# Get ETH rebalancing data from Excel

timestamp = datetime.date(2021,5,19)
excelfile = f"{filepath}/Leverage Tokens/BinanceLeverageToken.xlsx"

sushiup_excel = pd.read_excel(excelfile, sheet_name=f"{currency.capitalize()}Up", usecols = "A:H")
sushidown_excel = pd.read_excel(excelfile, sheet_name=f"{currency.capitalize()}Down", usecols = "A:H")

columns = ["BasketBefore", "BasketAfter", "LeverageBefore", "LeverageAfter", "TokensBefore"]


# Clean data and convert to float - why only on sushiup?
try:
    for col in columns:
        sushiup_excel[col]  = sushiup_excel[col].str.replace('\xa0', '', regex=True)
        sushiup_excel[col]  = sushiup_excel[col].str.replace('+', '', regex=True)
        sushiup_excel[col]  = sushiup_excel[col].str.replace(',', '', regex=True).astype(float)
except:
    pass

sushiup_excel = (sushiup_excel
         # .loc[lambda df: df.Time.dt.date == timestamp]
         .sort_values(["Time"])
         .reset_index(drop=True)
         )

sushidown_excel = (sushidown_excel
         # .loc[lambda df: df.Time.dt.date == timestamp]
         .sort_values(["Time"])
         .reset_index(drop=True)
         )


# merge Excel data to the prices
sushi_merged = (pd
        .merge_asof(sushi_merged, sushiup_excel[["Time", 'BasketAfter', 'TokensAfter']], left_on="timestamp", right_on="Time")
        .rename(columns={"BasketAfter": "BasketUP", 'TokensAfter': "nTokensUP"})
        )

sushi_merged = (pd
        .merge_asof(sushi_merged, sushidown_excel[["Time", 'BasketAfter', 'TokensAfter']], left_on="timestamp", right_on="Time")
        .rename(columns={"BasketAfter": "BasketDOWN", 'TokensAfter': "nTokensDOWN"})
        )


# Add leverageUP and leverageDOWN to sushi_merged
sushi_merged2 = (sushi_merged
             .assign(leverageUP   = lambda df: (df.BasketUP*df.sushi_price)/(df.nTokensUP*df.sushiup_price))
             .assign(leverageDOWN = lambda df: -(df.BasketDOWN*df.sushi_price)/(df.nTokensDOWN*df.sushidown_price))
             )



# --------------------------------------------------
#%% Limit Data to Given Time Window
#--------------------------------------------------

# Keep only data for hours to be analysed
plotstartS = datetime.datetime(2021,5,19, 12,0,0)
plotendS   = datetime.datetime(2021,5,19, 14,0,0)


indx = (sushi_merged2.timestamp>=plotstartS) & (sushi_merged2.timestamp<=plotendS)
sushi_merged3 = sushi_merged2.loc[indx]

#sushi_merged3.to_csv(f'{filepath}/sushi_merged3.csv')

