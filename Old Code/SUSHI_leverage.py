# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 17:08:37 2023

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


filepath = "D:/dissertation-data"
plots_path = "C:/Users/gianc/OneDrive/Uni/Sussex/Dissertation/Code/sushi_plots"
_ = update_plt_params(20)


# =============================================================================
# load trade data
# =============================================================================
#%%

if True: #Using trade data - change to False to run 'else' codeblock
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
    
else: # Using bookTicker data
    
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
# calculate NAV and leverage
# =============================================================================
#%%

df_merged2 = (df_merged
             .assign(leverageUP   = lambda df: (df.BastketUP*df.sushi_Price)/(df.nTokensUP*df.sushiup_Price))
             .assign(leverageDOWN = lambda df: -(df.BastketDOWN*df.sushi_Price)/(df.nTokensDOWN*df.sushidown_Price))
             )

# df_merged2.to_csv("Leverage.csv")


# =============================================================================
# plot results with Latency - new_leverage
# =============================================================================
#%%

start_time, end_time = 11, 20 # Start and end times for longer timeframe plot
start_timeS, end_timeS = 13, 15 # Start and end times for 'zoomed in' plot

plotstart = datetime.datetime(2021,5,19, start_time,0,0)
plotend   = datetime.datetime(2021,5,19, end_time,0,1)

plotstartS = datetime.datetime(2021,5,19, start_timeS,0,0)
plotendS   = datetime.datetime(2021,5,19, end_timeS,0,0)

specs = [{"variable": "leverageUP", "rebalancing": sushiup.Time, "color": "green", "byminute": [0], "plotstart": plotstart, "plotend": plotend, "name": f"UP_{start_time}_{end_time}"},
         {"variable": "leverageUP", "rebalancing": sushiup.Time, "color": "green", "byminute": [0,15,30,45], "plotstart": plotstartS, "plotend": plotendS, "name": f"UP_{start_timeS}_{end_timeS}"},
         {"variable": "leverageDOWN", "rebalancing": sushidown.Time, "color": "red", "byminute": [0], "plotstart": plotstart, "plotend": plotend, "name": f"DOWN_{start_time}_{end_time}"},
         {"variable": "leverageDOWN", "rebalancing": sushidown.Time, "color": "red", "byminute": [0,15,30,45], "plotstart": plotstartS, "plotend": plotendS, "name": f"DOWN_{start_timeS}_{end_timeS}"}
        ]

for spec in specs:
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,6), sharex=False, sharey=False)
    
    # select data
    indx = (df_merged2.timestamp.dt.time>=spec["plotstart"].time()) & (df_merged2.timestamp.dt.time<=spec["plotend"].time())
    x = df_merged2["timestamp"].loc[indx]
    y = df_merged2[spec["variable"]].loc[indx]
    y2= df_merged2["sushi_Latency"].loc[indx]
    rebal_times = spec["rebalancing"]
    rebal_times = rebal_times[(rebal_times>=spec["plotstart"]) & (rebal_times<=spec["plotend"])]
    

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(True)
    axes.spines['left'].set_visible(True)
    
    # plot the data
    label = "Token UP Leverage" if spec["variable"]=="leverageUP" else "Token DOWN Leverage" 
    ln1 = axes.plot(x,y, label=label+ ", left axis", color=spec["color"], marker=None, markersize = 5, ls = '-')
    axes.plot(x,4*np.ones(len(x)), color=spec["color"], marker=None, markersize = 5, ls = '--')
    axes.plot(x,1.25*np.ones(len(x)), color=spec["color"], marker=None, markersize = 5, ls = '--')
    
    ax2 = axes.twinx()
    ln3 = ax2.fill_between(x,0, y2, color="tab:pink", label="Latency (in 1000 secs), right axis", alpha = 0.2)
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes.xaxis.set_major_locator(MinuteLocator(byminute=spec["byminute"], interval=1))
    
    
    for i, xc in enumerate(rebal_times):
        if i==0:
            ln2 = axes.axvline(x=xc, color="black", label="Rebalancing times", ls="--", linewidth = 1)
        else:
            axes.axvline(x=xc, color="black", ls="--", linewidth = 1)
    
    # ask matplotlib for the plotted objects and their labels
    lines, labels = axes.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', frameon=False)
    
    fig = fig.get_figure()
    fig.savefig(f"{plots_path}/new_leverage/new_leverage_{spec['name']}.png", bbox_inches='tight', dpi=2000)
    plt.close(fig)
    
#%%

# =============================================================================
# plot results with Latency - new_leverage_return
# =============================================================================
#%%

start_time, end_time = 11, 20 # Start and end times for longer timeframe plot
start_timeS, end_timeS = 13, 15 # Start and end times for 'zoomed in' plot

plotstart = datetime.datetime(2021,5,19, start_time,0,0)
plotend   = datetime.datetime(2021,5,19, end_time,0,1)

plotstartS = datetime.datetime(2021,5,19, start_timeS,0,0)
plotendS   = datetime.datetime(2021,5,19, end_timeS,0,0)

df_merged2["sushi_return"] = df_merged2["sushi_Price"]/df_merged2["sushi_Price"].shift(1)-1


specs = [{"variable": "leverageUP", "return": "sushi_return", "rebalancing": sushiup.Time, "color": "green", "byminute": [0], "plotstart": plotstart, "plotend": plotend, "name": f"UP_{start_time}_{end_time}"},
         {"variable": "leverageUP", "return": "sushi_return", "rebalancing": sushiup.Time, "color": "green", "byminute": [0,15,30,45], "plotstart": plotstartS, "plotend": plotendS, "name": f"UP_{start_timeS}_{end_timeS}"},
         {"variable": "leverageDOWN", "return": "sushi_return", "rebalancing": sushidown.Time, "color": "red", "byminute": [0], "plotstart": plotstart, "plotend": plotend, "name": f"DOWN_{start_time}_{end_time}"},
         {"variable": "leverageDOWN", "return": "sushi_return", "rebalancing": sushidown.Time, "color": "red", "byminute": [0,15,30,45], "plotstart": plotstartS, "plotend": plotendS, "name": f"DOWN_{start_timeS}_{end_timeS}"}
        ]



for spec in specs:
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,6), sharex=False, sharey=False)
    
    # select data
    indx = (df_merged2.timestamp.dt.time>=spec["plotstart"].time()) & (df_merged2.timestamp.dt.time<=spec["plotend"].time())
    x = df_merged2["timestamp"].loc[indx]
    y = df_merged2[spec["variable"]].loc[indx]
    y2= df_merged2[spec["return"]].loc[indx]*100*(1 if spec["variable"]=="leverageUP" else -1)
    
    rebal_times = spec["rebalancing"]
    rebal_times = rebal_times[(rebal_times>=spec["plotstart"]) & (rebal_times<=spec["plotend"])]
    

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(True)
    axes.spines['left'].set_visible(True)
    
    label = "Token UP Leverage" if spec["variable"]=="leverageUP" else "Token DOWN Leverage" 
    # plot the data
    ln1 = axes.plot(x,y, label=label+", left axis", color=spec["color"], marker=None, linewidth = 1, ls = '-')
    axes.plot(x,4*np.ones(len(x)), color=spec["color"], marker=None, linewidth = 1, ls = '--')
    axes.plot(x,1.25*np.ones(len(x)), color=spec["color"], marker=None, linewidth = 1, ls = '--')
    
    ax2 = axes.twinx()
    ax2.plot(x,y2, label="SUSHI return (%), right axis", color="black", marker=None, linewidth = 1, ls = '-')
    ax2.plot(x,0*np.ones(len(x)), color="gray", marker=None, linewidth = 0.5, ls = '-')
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes.xaxis.set_major_locator(MinuteLocator(byminute=spec["byminute"], interval=1))
    
    for i, xc in enumerate(rebal_times):
        if i==0:
            ln2 = axes.axvline(x=xc, color="black", label="Rebalancing times", ls="--", linewidth = 1)
        else:
            axes.axvline(x=xc, color="black", ls="--", linewidth = 1)
    
    # ask matplotlib for the plotted objects and their labels
    lines, labels = axes.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', frameon=False)
    
    fig = fig.get_figure()
    fig.savefig(f"{plots_path}/new_leverage_return/new_leverage_return_{spec['name']}.png", bbox_inches='tight', dpi=2000)
    plt.close(fig)
    
#%%

# =============================================================================
# plot results with leveraged returns
# =============================================================================
#%%

start_time, end_time = 11, 20 # Start and end times for longer timeframe plot
start_timeS, end_timeS = 13, 15 # Start and end times for 'zoomed in' plot

plotstart = datetime.datetime(2021,5,19, start_time,0,0)
plotend   = datetime.datetime(2021,5,19, end_time,0,1)

plotstartS = datetime.datetime(2021,5,19, start_timeS,0,0)
plotendS   = datetime.datetime(2021,5,19, end_timeS,0,0)

df_merged2["sushi_return"] = df_merged2["sushi_Price"]/df_merged2["sushi_Price"].shift(1)-1


specs = [{"variable": "leverageUP", "return": "sushi_return", "rebalancing": sushiup.Time, "color": "green", "byminute": [0], "plotstart": plotstart, "plotend": plotend, "name": f"UP_{start_time}_{end_time}"},
         {"variable": "leverageUP", "return": "sushi_return", "rebalancing": sushiup.Time, "color": "green", "byminute": [0,15,30,45], "plotstart": plotstartS, "plotend": plotendS, "name": f"UP_{start_timeS}_{end_timeS}"},
         {"variable": "leverageDOWN", "return": "sushi_return", "rebalancing": sushidown.Time, "color": "red", "byminute": [0], "plotstart": plotstart, "plotend": plotend, "name": f"DOWN_{start_time}_{end_time}"},
         {"variable": "leverageDOWN", "return": "sushi_return", "rebalancing": sushidown.Time, "color": "red", "byminute": [0,15,30,45], "plotstart": plotstartS, "plotend": plotendS, "name": f"DOWN_{start_timeS}_{end_timeS}"}
        ]


for spec in specs:
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,6), sharex=False, sharey=False)
    
    # select data
    indx = (df_merged2.timestamp.dt.time>=spec["plotstart"].time()) & (df_merged2.timestamp.dt.time<=spec["plotend"].time())
    x = df_merged2["timestamp"].loc[indx]
    y = df_merged2[spec["variable"]].loc[indx]
    y2= df_merged2[spec["return"]].loc[indx]*100*y.shift(1)*(1 if spec["variable"]=="leverageUP" else -1)
    
    rebal_times = spec["rebalancing"]
    rebal_times = rebal_times[(rebal_times>=spec["plotstart"]) & (rebal_times<=spec["plotend"])]
    

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(True)
    axes.spines['left'].set_visible(True)
    
    label = "Token UP Leverage" if spec["variable"]=="leverageUP" else "Token DOWN Leverage" 
    # plot the data
    ln1 = axes.plot(x,y, label=label+", left axis", color=spec["color"], marker=None, linewidth = 1, ls = '-')
    axes.plot(x,4*np.ones(len(x)), color=spec["color"], marker=None, linewidth = 1, ls = '--')
    axes.plot(x,1.25*np.ones(len(x)), color=spec["color"], marker=None, linewidth = 1, ls = '--')
    
    ax2 = axes.twinx()
    ln2 = ax2.plot(x,y2, label="SUSHI Leveraged Return (%), right axis", color="black", marker=None,  linewidth = 1, ls = '-')
    ax2.plot(x,0*np.ones(len(x)), color="gray", marker=None, linewidth = 0.5, ls = '-')
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes.xaxis.set_major_locator(MinuteLocator(byminute=spec["byminute"], interval=1))
    
    # for i, xc in enumerate(rebal_times):
    #     if i==0:
    #         ln2 = axes.axvline(x=xc, color=spec["color"], label="Rebalancing times", linewidth = 2,alpha=0.1)
    #     else:
    #         axes.axvline(x=xc, color=spec["color"], linewidth = 2,alpha=0.2)
    
    # ask matplotlib for the plotted objects and their labels
    lines, labels = axes.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', frameon=False)
    
    fig = fig.get_figure()
    fig.savefig(f"{plots_path}/new_leverage_leveragedreturn/new_leverage_leveragedreturn_{spec['name']}.png", bbox_inches='tight', dpi=2000)
    plt.close(fig)
    


#%%

# new_leveraged_prices

start_time, end_time = 11, 17 # Start and end times for longer timeframe plot
start_timeS, end_timeS = 13, 15 # Start and end times for 'zoomed in' plot

plotstart = datetime.datetime(2021,5,19, start_time,0,0)
plotend   = datetime.datetime(2021,5,19, end_time,0,1)

plotstartS = datetime.datetime(2021,5,19, start_timeS,0,0)
plotendS   = datetime.datetime(2021,5,19, end_timeS,0,0)


df_merged2["sushi_return"] = df_merged2["sushi_Price"]/df_merged2["sushi_Price"].shift(1)-1


specs = [{"variable": "sushiup_Price", "legend": "UP Token Price, left axis", "return": "sushi_return", "rebalancing": sushiup.Time, "color": "green", "byminute": [0], "plotstart": plotstart, "plotend": plotend, "name": f"UP_{start_time}_{end_time}"},
         {"variable": "sushiup_Price", "legend": "UP Token Price, left axis", "return": "sushi_return", "rebalancing": sushiup.Time, "color": "green", "byminute": [0,15,30,45], "plotstart": plotstartS, "plotend": plotendS, "name": f"UP_{start_timeS}_{end_timeS}"},
         {"variable": "sushidown_Price", "legend": "DOWN Token Price, left axis", "return": "sushi_return", "rebalancing": sushidown.Time, "color": "red", "byminute": [0], "plotstart": plotstart, "plotend": plotend, "name": f"DOWN_{start_time}_{end_time}"},
         {"variable": "sushidown_Price", "legend": "DOWN Token Price, left axis", "return": "sushi_return", "rebalancing": sushidown.Time, "color": "red", "byminute": [0,15,30,45], "plotstart": plotstartS, "plotend": plotendS, "name": f"DOWN_{start_timeS}_{end_timeS}"}
        ]



for spec in specs:
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,6), sharex=False, sharey=False)
    
    # select data
    
    indx = (df_merged2.timestamp.dt.time>=spec["plotstart"].time()) & (df_merged2.timestamp.dt.time<=spec["plotend"].time())
    x = df_merged2.timestamp[indx]
    y = df_merged2[spec["variable"]].loc[indx]
    y2 = df_merged2["sushi_Latency"].loc[indx]
    
    ln1 = axes.plot(x,y, label=spec["legend"], color=spec["color"], marker=None,  linewidth = 1, ls = '-')

    axes.spines['top'].set_visible(True)
    axes.spines['right'].set_visible(True)
    axes.spines['bottom'].set_visible(True)
    axes.spines['left'].set_visible(True)
    
    # plot the data
    
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes.xaxis.set_major_locator(MinuteLocator(byminute=spec["byminute"], interval=1))
    
    ax2 = axes.twinx()
    ln3 = ax2.fill_between(x,0, y2, color="tab:pink", label="Latency (in 1000 secs), right axis", alpha = 0.2)
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes.xaxis.set_major_locator(MinuteLocator(byminute=spec["byminute"], interval=1))
    
    lines, labels = axes.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', frameon=False)
    
    
    fig = fig.get_figure()
    fig.savefig(f"{plots_path}/new_leveraged_prices/new_leverage_prices_{spec['name']}.png", bbox_inches='tight', dpi=2000)
    plt.close(fig)
    
#%%



#%% PLOT THE RAW PRICES AT TRADE LEVEL

# sushiupusdt
# sushidowndt
# sushiusdt

timeinterv = "1S"

    
sushiup1S = (sushiupusdt
              .assign(timestamp = lambda df: df.Event_time)
              .loc[:, ["timestamp", "Price"]]
              .sort_values("timestamp")
              .set_index("timestamp")
              .resample(timeinterv, closed="right", label="right")
              .last()
              .reset_index(drop=False)
              .rename(columns = lambda x: f"sushiup_{x}" if x != "timestamp" else x)
              )

sushidown1S = (sushidowndt
              .assign(timestamp = lambda df: df.Event_time)
              .loc[:, ["timestamp", "Price"]]
              .sort_values("timestamp")
              .set_index("timestamp")
              .resample(timeinterv, closed="right", label="right")
              .last()
              .reset_index(drop=False)
              .rename(columns = lambda x: f"sushidown_{x}" if x != "timestamp" else x)
              )

sushi1S = (sushiusdt
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

#%%

# new_leverage_All_prices

plotstartS = datetime.datetime(2021,5,19, 12,45,0)
plotendS   = datetime.datetime(2021,5,19, 13,15,0)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,6), sharex=False, sharey=False)

# select sushi data
indx = (sushi1S.timestamp.dt.time>=plotstartS.time()) & (sushi1S.timestamp.dt.time<=plotendS.time())
x1 = sushi1S.loc[indx, ["timestamp"]]
y1 = sushi1S.loc[indx, ["sushi_Price"]]

# select sushi_UP data
indx = (sushiup1S.timestamp.dt.time>=plotstartS.time()) & (sushiup1S.timestamp.dt.time<=plotendS.time())
x2 = sushiup1S.loc[indx, ["timestamp"]]
y2 = sushiup1S.loc[indx, ["sushiup_Price"]]

# select sushi data
indx = (sushidown1S.timestamp.dt.time>=plotstartS.time()) & (sushidown1S.timestamp.dt.time<=plotendS.time())
x3 = sushidown1S.loc[indx, ["timestamp"]]
y3 = sushidown1S.loc[indx, ["sushidown_Price"]]



axes.spines['top'].set_visible(True)
axes.spines['right'].set_visible(True)
axes.spines['bottom'].set_visible(True)
axes.spines['left'].set_visible(True)
axes.set_ylim([25, 375])
# plot the data
axes.plot(x1,y1/10, label="SUSHI Price (divided by 10), left axis", color="black", marker=None,  linewidth = 1, ls = '-')
axes.plot(x2,y2, label="Token UP Price, left axis", color="green", marker=None,  linewidth = 1, ls = '-')

axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axes.xaxis.set_major_locator(MinuteLocator(byminute=spec["byminute"], interval=1))

ax2 = axes.twinx()
ax2.plot(x3,y3*1000, label="Token DOWN Price (times 1000), right axis", color="red", marker=None,  linewidth = 1, ls = '-')
ax2.set_ylim([0.5, 3.25])
# axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# axes.xaxis.set_major_locator(MinuteLocator(byminute=spec["byminute"], interval=1))

lines, labels = axes.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right', frameon=False)


fig = fig.get_figure()
fig.savefig(f"{plots_path}/new_leverage_All_prices.png", bbox_inches='tight', dpi=2000)
plt.close(fig)


#%%

# new_leverage_All_latency

plotstartS = datetime.datetime(2021,5,19, 12,45,0)
plotendS   = datetime.datetime(2021,5,19, 14,15,0)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,6), sharex=False, sharey=False)

# select sushi data
indx = (sushi1M.timestamp.dt.time>=plotstartS.time()) & (sushi1M.timestamp.dt.time<=plotendS.time())
x1 = sushi1M.loc[indx, ["timestamp"]]
y1 = sushi1M.loc[indx, ["sushi_Latency"]]

# select sushi_UP data
indx = (sushiup1M.timestamp.dt.time>=plotstartS.time()) & (sushiup1M.timestamp.dt.time<=plotendS.time())
x2 = sushiup1M.loc[indx, ["timestamp"]]
y2 = sushiup1M.loc[indx, ["sushiup_Latency"]]

# select sushi data
indx = (sushidown1M.timestamp.dt.time>=plotstartS.time()) & (sushidown1M.timestamp.dt.time<=plotendS.time())
x3 = sushidown1M.loc[indx, ["timestamp"]]
y3 = sushidown1M.loc[indx, ["sushidown_Latency"]]



axes.spines['top'].set_visible(True)
axes.spines['right'].set_visible(True)
axes.spines['bottom'].set_visible(True)
axes.spines['left'].set_visible(True)

# plot the data

axes.plot(x2,y2, label="Token UP Latency (in 1000 secs), right axis", color="green", marker=None,  linewidth = 1, ls = '-')
axes.plot(x3,y3, label="Token DOWN Latency (in 1000 secs), right axis", color="red", marker=None,  linewidth = 1, ls = '-')
axes.plot(x1,y1, label="SUSHI Latency (in 1000 secs), left axis", color="black", marker=None,  linewidth = 1, ls = '-')

axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axes.xaxis.set_major_locator(MinuteLocator(byminute=[15,45], interval=1))

# ax2 = axes.twinx()
# ax2.plot(x2,y2, label="Token UP Latency (in 1000 secs), right axis", color="green", marker=None,  linewidth = 1, ls = '-')
# ax2.plot(x3,y3, label="Token DOWN Latency (in 1000 secs), right axis", color="red", marker=None,  linewidth = 1, ls = '-')


# axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# axes.xaxis.set_major_locator(MinuteLocator(byminute=spec["byminute"], interval=1))

lines, labels = axes.get_legend_handles_labels()
axes.legend(lines, labels, loc='upper right', frameon=False)


fig = fig.get_figure()
fig.savefig(f"{plots_path}/new_leverage_All_latency.png", bbox_inches='tight', dpi=2000)
plt.close(fig)



