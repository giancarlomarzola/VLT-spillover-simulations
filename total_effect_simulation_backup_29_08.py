# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:44:06 2023

@author: gm399
"""

from orderbook import make_orderbook, plot_orderbook
from simulation_functions import load_raw_data, resample_data, reformat_data, token_characteristics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import datetime
import matplotlib.ticker as ticker


# Needed for plot
import matplotlib.dates as mdates
from matplotlib.dates import HourLocator, MonthLocator, YearLocator, MinuteLocator


# PC filepath
filepath = "C:/Users/gianc/OneDrive/Uni/Sussex/Dissertation/Code/dissertation-data"
plot_path = "C:/Users/gianc/OneDrive/Uni/Sussex/Dissertation/Plots Orderbooks"

#Laptop filepath
#filepath = "C:/Users/Giancarlo/OneDrive/Uni/Sussex/Dissertation/Code/dissertation-data"

token = 'sushi'

# --------------------------------------------------
#%% Import Token Data
#--------------------------------------------------

if True: # Use minute data
    # Load ETH data files
    sushiup, sushiup_     = load_raw_data(filepath, "binance", f"{token}upusdt", "2021-05-19", "trade")
    sushidown, sushidown_ = load_raw_data(filepath, "binance", f"{token}downusdt", "2021-05-19", "trade")
    sushi, sushi_         = load_raw_data(filepath, "binance-futures", f"{token}usdt", "2021-05-19", "trade")
    
    # Reduce data to minute instead of ms
    sushiup_minute   = resample_data(sushiup, 'sushiup', '1T')
    sushidown_minute = resample_data(sushidown, 'sushidown', '1T')
    sushi_minute     = resample_data(sushi, 'sushi', '1T')
    
    # merge the minute data to one df (merge)
    sushi_merged = reduce(lambda left, right: pd.merge(left, right, on="timestamp"), 
                          [sushiup_minute, sushidown_minute, sushi_minute])

else: # Use all data, rather than minute
        # Load ETH data files
    sushiup, sushiup_     = load_raw_data(filepath, "binance", f"{token}upusdt", "2021-05-19", "trade")
    sushidown, sushidown_ = load_raw_data(filepath, "binance", f"{token}downusdt", "2021-05-19", "trade")
    sushi, sushi_         = load_raw_data(filepath, "binance-futures", f"{token}usdt", "2021-05-19", "trade")
    
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

sushiup_excel = pd.read_excel(excelfile, sheet_name=f"{token.capitalize()}Up", usecols = "A:H")
sushidown_excel = pd.read_excel(excelfile, sheet_name=f"{token.capitalize()}Down", usecols = "A:H")

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


# --------------------------------------------------
#%% ORDERBOOK

# General orderbook params
tick_size            = 0.00005
max_deviation        = 12
beta, alpha          = tick_size*5, tick_size*5 # spread from mid to best bid (beta) and mid to best ask (alpha)

# Shallow and Narrow
worst_bid, worst_ask = 1, 1 # width
bid_depth            = 5_000_000/sushi_merged3["sushi_price"].iloc[0]
ask_depth            = 5_000_000/sushi_merged3["sushi_price"].iloc[0]

orderbook_sn = make_orderbook(alpha, beta, worst_ask, worst_bid, ask_depth, bid_depth, 
                                 max_deviation, tick_size)
plot_orderbook(orderbook_sn)#, plot_path, 'sn')


# Shallow and Wide - Mimics SUSHI
worst_bid, worst_ask = 10, 10 # width
bid_depth            = 5_000_000/sushi_merged3["sushi_price"].iloc[0]
ask_depth            = 5_000_000/sushi_merged3["sushi_price"].iloc[0]

orderbook_sw = make_orderbook(alpha, beta, worst_ask, worst_bid, ask_depth, bid_depth, 
                                 max_deviation, tick_size)
plot_orderbook(orderbook_sw)#, plot_path, 'sw')


# Deep and Narrow - Mimics ETH
worst_bid, worst_ask = 1, 1 # width 
bid_depth            = 50_000_000/sushi_merged3["sushi_price"].iloc[0] 
ask_depth            = 50_000_000/sushi_merged3["sushi_price"].iloc[0] # in tokens (USDT/price)

orderbook_dn = make_orderbook(alpha, beta, worst_ask, worst_bid, ask_depth, bid_depth, 
                               max_deviation, tick_size*20)
plot_orderbook(orderbook_dn)#, plot_path, 'dn')


# Deep and Wide
worst_bid, worst_ask = 10, 10 # width
bid_depth            = 50_000_000/sushi_merged3["sushi_price"].iloc[0]
ask_depth            = 50_000_000/sushi_merged3["sushi_price"].iloc[0]

orderbook_dw = make_orderbook(alpha, beta, worst_ask, worst_bid, ask_depth, bid_depth, 
                                 max_deviation, tick_size)
plot_orderbook(orderbook_dw)#, plot_path, 'dw')



#orderbooks = [orderbook_sn, orderbook_sw, orderbook_dn, orderbook_dw]

# --------------------------------------------------
#%% Token Simulations
#--------------------------------------------------

# Parameters (same for up and down)
lambda_target = 3 # Only used when rebalance = 'target'
lambda_up     = 4 # upper boundary
lambda_down   = 1.25 # lower boundary
#lambda_0      = 3.5 # Leverage at start
rebalance     = "boundary" # either 'boundary' or 'target'

underlying_price = np.array(sushi_merged3["sushi_price"].values)

# UP parameters
up_investment = sushi_merged3["nTokensUP"].iloc[0] * sushi_merged3["sushiup_price"].iloc[0]
#up_investment  = sushi_merged3["sushi_price"].iloc[0]
up_lambda_0    = sushi_merged3["leverageUP"].iloc[0]

# DOWN Parameters
#down_investment = 100_000#125_716_067.04 # Number of SUSHIDOWN issued on 19.05.2021; number kept changing
down_investment = sushi_merged3["nTokensDOWN"].iloc[0] * sushi_merged3["sushidown_price"].iloc[0]
#down_investment  = -sushi_merged3["BasketDOWN"].iloc[0] * sushi_merged3["sushi_price"].iloc[0]
down_lambda_0    = sushi_merged3["leverageDOWN"].iloc[0]

sn = token_characteristics(underlying_price, lambda_target, lambda_up, lambda_down,
                                             up_investment, up_lambda_0, down_investment, down_lambda_0,
                                             rebalance, orderbook_sn)

sw = token_characteristics(underlying_price, lambda_target, lambda_up, lambda_down,
                                             up_investment, up_lambda_0, down_investment, down_lambda_0,
                                             rebalance, orderbook_sw)

dw = token_characteristics(underlying_price, lambda_target, lambda_up, lambda_down,
                                             up_investment, up_lambda_0, down_investment, down_lambda_0,
                                             rebalance, orderbook_dw)

dn = token_characteristics(underlying_price, lambda_target, lambda_up, lambda_down,
                                             up_investment, up_lambda_0, down_investment, down_lambda_0,
                                             rebalance, orderbook_dn)
'''
simulations = {}
simulation_names = ['sn', 'sw', 'dn', 'dw']

for i, book in enumerate(orderbooks):
    sim_name = simulation_names[i]  # Create a unique result name
    simulations[sim_name] = token_characteristics(underlying_price, lambda_target, lambda_up, lambda_down,
                                                 up_investment, up_lambda_0, down_investment, down_lambda_0,
                                                 rebalance, book)
'''
# --------------------------------------------------
#%% Simulations Plots
#--------------------------------------------------

# Plot font sizes
axis_titles = 18
legends = 14

# Plotting market price v simulated price with rebalances
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), dpi=300)
x = sushi_merged3.timestamp
market_price = underlying_price
sn_price     = underlying_price*sn['price_multiplier']
sw_price     = underlying_price*sw['price_multiplier']
dn_price     = underlying_price*dn['price_multiplier']
dw_price     = underlying_price*dw['price_multiplier']
label0 = "actual price"
label1 = "simulated price - Small Narrow"
label2 = "simulated price - Small Wide"
label3 = "simulated price - Deep Narrow"
label4 = "simulated price - Deep Wide"

ax.plot(x, market_price, color='k', label=label0)
ax.plot(x, sn_price, color='r', ls='--', label=label1)
ax.plot(x, sw_price, color='g', ls='--',label=label2)
ax.plot(x, dn_price, color='b', ls='--',label=label3)
ax.plot(x, dw_price, color='orange', ls='--',label=label4)

# format the x-axis
plt.xlabel('Timestamp', fontsize=axis_titles)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(MinuteLocator(byminute=[0, 30], interval=1))
ax.tick_params(axis='both', which='major', labelsize=16)

# format the y-axis
plt.ylabel('SUSHI Price (USDT)', fontsize=axis_titles)

# Legend
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='lower left', frameon=False, fontsize=legends)
fig = fig.get_figure()
#fig.savefig("C:/Users/gianc/OneDrive/Uni/Sussex/Dissertation/Plots Model2/model2_SUSHI_price.png", bbox_inches='tight')#, dpi=300)
plt.show()


# Plotting UP and DOWN token NAVs
for token in ['UP', 'DOWN']:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), dpi=300)
    x = sushi_merged3.timestamp
    market_nav = sushi_merged3[f'sushi{token.lower()}_price']*sushi_merged3[f'nTokens{token}']
    sn_up_nav  = sn[f'v_{token.lower()}']
    sw_up_nav  = sw[f'v_{token.lower()}']
    dn_up_nav  = dn[f'v_{token.lower()}']
    dw_up_nav  = dw[f'v_{token.lower()}']
    label0 = "Actual NAV"
    label1 = "Shallow Narrow"
    label2 = "Shallow Wide"
    label3 = "Deep Narrow"
    label4 = "Deep Wide"
    
    ax.plot(x, market_nav, color='k', label=label0)
    ax.plot(x, sn_up_nav, color='r', ls='--', label=label1)
    ax.plot(x, sw_up_nav, color='g', ls='--',label=label2)
    ax.plot(x, dn_up_nav, color='b', ls='--',label=label3)
    ax.plot(x, dw_up_nav, color='orange', ls='--',label=label4)
    
    # format the x-axis
    plt.xlabel('Timestamp', fontsize=axis_titles)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(MinuteLocator(byminute=[0, 30], interval=1))
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # format the y-axis
    plt.ylabel(f'SUSHI{token} NAV (Million USDT)', fontsize=axis_titles)
    #plt.ylim(-1_000_000,41_000_000)
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/1000000), ',')))
    
    # Legend
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left', frameon=False, fontsize=legends)
    
    plt.show()

token = 'up'
fig, ax = plt.subplots(figsize=(12,6), dpi=300)
x = sushi_merged3.timestamp
y1 = sn[f'lambdast_{token}']
y2 = sw[f'lambdast_{token}']
y3 = dn[f'lambdast_{token}']
y4 = dw[f'lambdast_{token}']
ax.plot(x, y1, color='r', ls='--', label=f'Lambda_star {token.upper()} sn')
ax.plot(x, y2, color='g', ls='--', label=f'Lambda_star {token.upper()} sw')
ax.plot(x, y3, color='b', ls='--', label=f'Lambda_star {token.upper()} dn')
ax.plot(x, y4, color='orange', ls='--', label=f'Lambda_star {token.upper()} dw')
plt.ylim(-0.5, 10)
ax.legend()
plt.show()

#%%
'''
test['price_multiplier'].loc[:117].plot(label='Price Multiplier')
plt.ylim(0, 2)
plt.legend()
plt.show()

#test['orderbook_effect'].plot()

test['v_up'].plot(color='green', label='UP')
test['v_down'].plot(color='red', label='DOWN')
plt.legend(loc='upper right', frameon=False)


for book in orderbooks:
    test = token_characteristics(underlying_price, lambda_target, lambda_up, lambda_down,
                          up_investment, up_lambda_0, down_investment, down_lambda_0,
                          rebalance, book)

    # Plotting market price v simulated price with rebalances
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    x = sushi_merged3.timestamp
    y1 = underlying_price
    y2 = underlying_price*test['price_multiplier']
    label1 = "actual price"
    label2 = "simulated price"
    ax.plot(x, y1, color='g', label=label1)
    ax.plot(x, y2, color='r', label=label2)
    
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper right', frameon=False)
    
    plt.show()
    
    test['price_multiplier'].loc[:117].plot(label='Price Multiplier')
    plt.ylim(0, 2)
    plt.legend()
    plt.show()
    
    #test['orderbook_effect'].plot()
    
    test['v_up'].plot(color='green', label='UP')
    test['v_down'].plot(color='red', label='DOWN')
    plt.legend(loc='upper right', frameon=False)
'''