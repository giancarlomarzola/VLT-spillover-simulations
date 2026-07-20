
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:52:36 2023

@author: ak486 adapted by gm399
"""
import sys
# sys.path is a list of absolute path strings
sys.path.append("C:/Users/gianc/OneDrive/Uni/Sussex/Dissertation/Code/andreas_python_new")

from misc import up_down_token_characteristics, load_raw_data, update_plt_params
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, MonthLocator, YearLocator, MinuteLocator
import matplotlib.dates as mdates
import datetime
from functools import reduce


filepath = "C:/Users/gianc/OneDrive/Uni/Sussex/Dissertation/Code/dissertation-data"
plot_path = "C:/Users/gianc/OneDrive/Uni/Sussex/Dissertation/Code/plots_sushi_andreas_code"
_ = update_plt_params(20)


# =============================================================================
# load trade data
# =============================================================================
#%%

if True:
    sushiupusdt, sushiupusdt_     = load_raw_data(filepath, "binance", "sushiupusdt", "2021-05-19", "trade")
    sushidowndt, sushidowndt_     = load_raw_data(filepath, "binance", "sushidownusdt", "2021-05-19", "trade")
    sushiusdt, sushiusdt_         = load_raw_data(filepath, "binance-futures", "sushiusdt", "2021-05-19", "trade")
    
    
    timeinterv = "1T"
    
    sushiup1M = (sushiupusdt
                  .assign(timestamp = lambda df: df.Event_time)
                  .assign(Latency = lambda df: ((df.Event_time-df.Trade_time).dt.total_seconds()).astype(int)/1000)
                  .loc[:, ["timestamp", "Price", "Latency"]]
                  .sort_values("timestamp")
                  .set_index("timestamp")
                  .resample(timeinterv, closed="right", label="right")
                  .last()
                  .reset_index(drop=False)
                  .rename(columns = lambda x: f"sushiup_{x}" if x != "timestamp" else x)
                  )
    
    sushidown1M = (sushidowndt
                  .assign(timestamp = lambda df: df.Event_time)
                  .assign(Latency = lambda df: ((df.Event_time-df.Trade_time).dt.total_seconds()).astype(int)/1000)
                  .loc[:, ["timestamp", "Price", "Latency"]]
                  .sort_values("timestamp")
                  .set_index("timestamp")
                  .resample(timeinterv, closed="right", label="right")
                  .last()
                  .reset_index(drop=False)
                  .rename(columns = lambda x: f"sushidown_{x}" if x != "timestamp" else x)
                  )
    
    sushi1M = (sushiusdt
                 .assign(timestamp = lambda df: df.Event_time)
                 .assign(Latency = lambda df: ((df.Event_time-df.Trade_time).dt.total_seconds()).astype(int)/1000)
                 .loc[:, ["timestamp", "Price", "Latency"]]
                 .sort_values("timestamp")
                 .set_index("timestamp")
                 .resample(timeinterv, closed="right", label="right")
                 .last()
                 .reset_index(drop=False)
                 .rename(columns = lambda x: f"sushi_{x}" if x != "timestamp" else x)
                 )
    
    # merge the prices
    df_merged = reduce(lambda left, right: pd.merge(left, right, on="timestamp"), [sushiup1M, sushidown1M, sushi1M])
    
else:
    
    sushiupusdt, sushiupusdt_     = load_raw_data(filepath, "binance", "sushiupusdt", "2021-05-19", "bookTicker")
    sushidowndt, sushidowndt_     = load_raw_data(filepath, "binance", "sushidownusdt", "2021-05-19", "bookTicker")
    sushiusdt, sushiusdt_         = load_raw_data(filepath, "binance-futures", "sushiusdt", "2021-05-19", "bookTicker")
    
    
    timeinterv = "1T"
    
    sushiup1M = (sushiupusdt
                  .assign(timestamp = lambda df: pd.to_datetime(df.local_timestamp))
                  .sort_values(["local_timestamp"])
                  .assign(Price = lambda df: 0.5*df["best_bid"] + 0.5*df["best_ask"])
                  .loc[:, ["timestamp", "Price"]]
                  .sort_values("timestamp")
                  .set_index("timestamp")
                  .resample(timeinterv, closed="right", label="right")
                  .last()
                  .reset_index(drop=False)
                  .rename(columns = lambda x: f"sushiup_{x}" if x != "timestamp" else x)
                  )
    
    sushidown1M = (sushidowndt
                  .assign(timestamp = lambda df: pd.to_datetime(df.local_timestamp))
                  .sort_values(["local_timestamp"])
                  .assign(Price = lambda df: 0.5*df["best_bid"] + 0.5*df["best_ask"])
                  .loc[:, ["timestamp", "Price"]]
                  .sort_values("timestamp")
                  .set_index("timestamp")
                  .resample(timeinterv, closed="right", label="right")
                  .last()
                  .reset_index(drop=False)
                  .rename(columns = lambda x: f"sushidown_{x}" if x != "timestamp" else x)
                  )
    
    sushi1M = (sushiusdt
                 .assign(timestamp = lambda df: pd.to_datetime(df.local_timestamp))
                 .sort_values(["local_timestamp"])
                 .assign(Price = lambda df: 0.5*df["best_bid"] + 0.5*df["best_ask"])
                 .loc[:, ["timestamp", "Price"]]
                 .sort_values("timestamp")
                 .set_index("timestamp")
                 .resample(timeinterv, closed="right", label="right")
                 .last()
                 .reset_index(drop=False)
                 .rename(columns = lambda x: f"sushi_{x}" if x != "timestamp" else x)
                 )
    
    # merge the prices
    df_merged = reduce(lambda left, right: pd.merge(left, right, on="timestamp"), [sushiup1M, sushidown1M, sushi1M])

# =============================================================================
# Download the rebalancing data
# =============================================================================
#%%

timestamp = datetime.date(2021,5,19)
filename = f"{filepath}/Leverage Tokens/BinanceLeverageToken.xlsx"
sushiup = pd.read_excel(filename, sheet_name="SushiUp", usecols = "A:H")

columns = ["BasketBefore", "BasketAfter", "LeverageBefore", "LeverageAfter", "TokensBefore"]
for col in columns:
    sushiup[col] = sushiup[col].str.replace('\xa0', '')
    sushiup[col]  = sushiup[col].str.replace('+', '')
    sushiup[col]  = sushiup[col].str.replace(',', '').astype(float)

sushidown = pd.read_excel(filename, sheet_name="SushiDown", usecols = "A:H")

sushiup = (sushiup
         # .loc[lambda df: df.Time.dt.date == timestamp]
         .sort_values(["Time"])
         .reset_index(drop=True)
         )

sushidown = (sushidown
         # .loc[lambda df: df.Time.dt.date == timestamp]
         .sort_values(["Time"])
         .reset_index(drop=True)
         )

# merge the data to the prices
df_merged = (pd
        .merge_asof(df_merged, sushiup[["Time", 'BasketAfter', 'TokensAfter']], left_on="timestamp", right_on="Time")
        .rename(columns={"BasketAfter": "BastketUP", 'TokensAfter': "nTokensUP"})
        )

df_merged = (pd
        .merge_asof(df_merged, sushidown[["Time", 'BasketAfter', 'TokensAfter']], left_on="timestamp", right_on="Time")
        .rename(columns={"BasketAfter": "BastketDOWN", 'TokensAfter': "nTokensDOWN"})
        )

# =============================================================================
# calculate NAV and leverage, and simulate the tokens using rebalancing rules
# =============================================================================
#%%

df_merged2 = (df_merged
             .assign(leverageUP   = lambda df: (df.BastketUP*df.sushi_Price)/(df.nTokensUP*df.sushiup_Price))
             .assign(leverageDOWN = lambda df: -(df.BastketDOWN*df.sushi_Price)/(df.nTokensDOWN*df.sushidown_Price))
             )


# =============================================================================
# plot results 
# =============================================================================
#%%

plotstartS = datetime.datetime(2021,5,19, 12,0,0)
plotendS   = datetime.datetime(2021,5,19, 14,0,0)

indx = (df_merged2.timestamp>=plotstartS) & (df_merged2.timestamp<=plotendS)
df_merged3 = df_merged2.loc[indx]



lambda_target = 0.0
lambda_up = 4
lambda_down = 1.25
lambda_0 = 3.5
rebalance = "boundary"

investment = df_merged3["sushiup_Price"].iloc[0]
lambda_0   = df_merged3["leverageUP"].iloc[0]
price = np.array(df_merged3["sushi_Price"].values)

tk_up     = up_down_token_characteristics(price, investment, lambda_target, lambda_up, lambda_down, lambda_0, rebalance)
tk_up_unc = up_down_token_characteristics(price, investment, lambda_target, 100_000, -100_000, lambda_0, rebalance)

investment = df_merged3["sushidown_Price"].iloc[0]
lambda_0   = df_merged3["leverageDOWN"].iloc[0]
price = np.array(df_merged3["sushi_Price"].values)

tk_down     = up_down_token_characteristics(price, investment, lambda_target, lambda_up, lambda_down, lambda_0, rebalance)
tk_down_unc = up_down_token_characteristics(price, investment, lambda_target, 100_000, -100_000, lambda_0, rebalance)

for token in ["UP", "DOWN"]:
    

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,6), sharex=False, sharey=False)
    
    axes.spines['top'].set_visible(True)
    axes.spines['right'].set_visible(True)
    axes.spines['bottom'].set_visible(True)
    axes.spines['left'].set_visible(True)
    
    label = f"Token {token} market price"
    label2 = f"Token {token} model price (with resets)"
    label3 = f"Token {token} model price (without resets)"
    x = df_merged3.timestamp
    y = df_merged3[f"sushi{token.lower()}_Price"]
    y2 = df_merged3[f"leverage{token}"]
    vt = tk_up.v_up if token=="UP" else  tk_down.v_down
    vt_unc = tk_up_unc.v_up if token=="UP" else  tk_down_unc.v_down
    
    lambdat = tk_up.lambda_up if token=="UP" else  tk_down.lambda_down
    color = "green" if token=="UP" else "red"
    
    # plot the data
    axes.plot(x,y, label=label,  color=color, marker=None, linewidth = 1, ls = '-')
    axes.plot(x,vt, label=label2, color=color, marker=None, linewidth = 1, ls = '--')
    axes.plot(x,vt_unc, label=label3, color=color, marker=None, linewidth = 1, ls = '-.')
    # format the x-axis
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes.xaxis.set_major_locator(MinuteLocator(byminute=[0, 30], interval=1))
    
    # # ask matplotlib for the plotted objects and their labels
    lines, labels = axes.get_legend_handles_labels()
    axes.legend(lines, labels, loc='lower left', frameon=False)
    
    fig = fig.get_figure()
    fig.savefig(f"{plot_path}/model1_SUSHI{token}_price.png", bbox_inches='tight', dpi=300)
    # plt.close(fig)
    



#%%
