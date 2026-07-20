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
from matplotlib.dates import MinuteLocator # HourLocator, MonthLocator, YearLocator, 


# PC filepath
filepath = "C:/Users/gianc/OneDrive/Uni/Sussex MSc/Dissertation/Code/dissertation-data"
plot_path = "C:/Users/gianc/OneDrive/Uni/Sussex/Dissertation/Plots"

#Laptop filepath
#filepath = "C:/Users/Giancarlo/OneDrive/Uni/Sussex/Dissertation/Code/dissertation-data"

currency = 'sushi' # all lowercase
minute_data = True

# Needed to save plots in correct location
model2_path, full = 'Model 2', '' # to save plots in correct location
if not minute_data:
    model2_path, full = 'Model 2 - Full Data', '_full'
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


# --------------------------------------------------
#%% ORDERBOOK

# General orderbook params
tick_size            = 0.001 # in %
max_deviation        = 12 # in %

# Other params with options
initial_price = sushi_merged3["sushi_price"].iloc[0]
depth_options = {'shallow':5_000_000/initial_price, # in USD / price = tokens
                 'deep':50_000_000/initial_price}

width_options = {'narrow':1,  #in %
                 'wide':10}

spread_options = {'tight':0.005, # in %
                  'broad':0.5}


orderbooks = {'no_book':None}

# Loop through all options and generate orderbook for each combination
for depth_level in depth_options:
    depth = depth_options[depth_level]
    depth_initial = depth_level[0]
    for width_level in width_options:
        width = width_options[width_level]
        width_initial = width_level[0]
        for spread_level in spread_options:
            spread = spread_options[spread_level]
            spread_initial = spread_level[0]
            b_name = f'{depth_initial}{width_initial}{spread_initial}'
            orderbooks[b_name] = make_orderbook(spread, spread, width, width, depth, depth, 
                                         max_deviation, tick_size)

# Plot all combinations
for book in orderbooks:
    try:
        plot_orderbook(orderbooks[book], f'{plot_path}/orderbooks', book)
    except:
        pass

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
up_lambda_0    = sushi_merged3["leverageUP"].iloc[0]

# DOWN Parameters
down_investment = sushi_merged3["nTokensDOWN"].iloc[0] * sushi_merged3["sushidown_price"].iloc[0]
down_lambda_0    = sushi_merged3["leverageDOWN"].iloc[0]


simulations = {}

# Add simulation without rebalances - this is same for all books so only needed once
simulations['no_balance'] = token_characteristics(underlying_price, lambda_target, 100_000, -100_000,
                                                  up_investment, up_lambda_0, down_investment, down_lambda_0,
                                                  rebalance)

# Add simulations with all types of orderbooks (including no orderbook)
for book in orderbooks:
    simulations[book] = token_characteristics(underlying_price, lambda_target, lambda_up, lambda_down,
                                              up_investment, up_lambda_0, down_investment, down_lambda_0,
                                              rebalance, orderbooks[book])





#--------------------------------------------------
#%% Plotting Futures Price
#--------------------------------------------------

# Plot font sizes
axis_titles = 18
legends = 12

# Label options
label_options = {'no_book':'No Orderbook',
                 'no_balance':'No Rebalancing',
                 'snt':'Shallow Narrow Tight', 
                 'snb':'Shallow Narrow Broad', 
                 'swt':'Shallow Wide Tight', 
                 'swb':'Shallow Wide Broad', 
                 'dnt':'Deep Narrow Tight', 
                 'dnb':'Deep Narrow Broad', 
                 'dwt':'Deep Wide Tight', 
                 'dwb': 'Deep Wide Broad'}

# Plotting market price v simulated price with rebalances
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), dpi=600)
x = sushi_merged3.timestamp

for combo in simulations:
    label = label_options[combo]
    # Exception for no orderbook and no rebalance
    if combo == 'no_book':
        y = underlying_price*simulations[combo]['price_multiplier']
        ax.plot(x, y, color='b', ls='--', label=label)
    elif combo == 'no_balance':
        y = underlying_price*simulations[combo]['price_multiplier']
        ax.plot(x, y, color='b', ls='-', alpha=0.5, label=label)
    else:
        color = 'r' if 's' in combo else 'g' # decide colour based on depth
        alpha = 0.3 if 'n' in combo else 1.0 # same as above but lighter if 't' in combo else darker
        ls = '-' if 't' in combo else '--' # decide line style based on width
        
        y =  underlying_price*simulations[combo]['price_multiplier']
        ax.plot(x, y, color=color, ls=ls, alpha=alpha, label=label)

# Plotting market price - no simulations of any kind
market_price = underlying_price
ax.plot(x, market_price, color='k', label='Observed Price')

# format the x-axis
plt.xlabel('Timestamp', fontsize=axis_titles)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(MinuteLocator(byminute=[0, 30], interval=1))
ax.tick_params(axis='both', which='major', labelsize=16)

# format the y-axis
plt.ylabel(f'{currency.upper()} Price (USDT)', fontsize=axis_titles)

# Legend
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='lower left', frameon=False, fontsize=legends)
fig = fig.get_figure()
fig.savefig(f'{plot_path}/{model2_path}/model2{full}_{currency.upper()}_price', bbox_inches='tight')#, dpi=300)
plt.show()

#%%
# Plotting UP and DOWN token NAVs
for token in ['UP', 'DOWN']:
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), dpi=300)
    x = sushi_merged3.timestamp
    
    for combo in simulations:
        label = label_options[combo]
        # Exception for no orderbook and no rebalancing
        if combo == 'no_book':
            y = simulations[combo][f'v_{token.lower()}']
            ax.plot(x, y, color='b', ls='--', label=label)
        elif combo == 'no_balance':
            y = simulations[combo][f'v_{token.lower()}']
            ax.plot(x, y, color='b', ls='-', alpha=0.5, label=label)
        else:
            # Format based on parameters
            color = 'r' if 's' in combo else 'g' # decide colour based on depth
            alpha = 0.3 if 'n' in combo else 1.0 # same as above but lighter if 't' in combo else darker
            ls = '-' if 't' in combo else '--' # decide line style based on width
            y = simulations[combo][f'v_{token.lower()}']
            ax.plot(x, y, color=color, ls=ls, alpha=alpha, label=label)
    
    # Add market price (do this after so it's plotted on top)
    market_nav = sushi_merged3[f'sushi{token.lower()}_price']*sushi_merged3[f'nTokens{token}']
    ax.plot(x, market_nav, color='k', label='Observed NAV')
    
    # format the x-axis
    plt.xlabel('Timestamp', fontsize=axis_titles)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")
                                 )
    ax.xaxis.set_major_locator(MinuteLocator(byminute=[0, 30], interval=1))
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # format the y-axis
    plt.ylabel(f'{currency.upper()}{token} NAV (Million USDT)', fontsize=axis_titles)
    #plt.ylim(-1_000_000,81_000_000)
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/1000000), ',')))
    
    # Legend
    lines, labels = ax.get_legend_handles_labels()
    loc = 'lower left' if token == 'UP' else 'upper left'
    ax.legend(lines, labels, loc=loc, frameon=False, fontsize=legends)
    fig = fig.get_figure()
    fig.savefig(f'{plot_path}/{model2_path}/model2{full}_{currency.upper()}{token}_NAV', bbox_inches='tight')
    plt.show()

#%% Plot Lambda star
'''
for token in ['up', 'down']:
    sign = 1 if token == 'up' else -1
    fig, ax = plt.subplots(figsize=(12,6), dpi=100)
    x = sushi_merged3.timestamp
    for combo in simulations:
        label = label_options[combo]
        if combo == 'no_book':
            y = sign * simulations[combo][f'lambdast_{token}']
            ax.plot(x, y, color='k', ls='--', label=label)
        elif combo == 'no_balance':
            y = sign * simulations[combo][f'lambdast_{token}']
            ax.plot(x, y, color='k', ls='--', alpha=0.5, label=label)
        else:
            color = 'r' if 's' in combo else 'g' # decide colour based on depth
            alpha = 0.3 if 'n' in combo else 1.0 # same as above but lighter if 't' in combo else darker
            ls = '-' if 't' in combo else '--' # decide line style based on width
            y = sign * simulations[combo][f'lambdast_{token}']
            ax.plot(x, y, color=color, ls=ls, alpha=alpha, label=label)
    
    # Leverage boundaries
    plt.axhline(y=sign*4, color='k', linestyle='--', linewidth=1) # upper
    plt.axhline(y=sign*1.25, color='k', linestyle='--', linewidth=1) # lower
    
    # X axis
    plt.xlabel('Timestamp', fontsize=axis_titles)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(MinuteLocator(byminute=[0, 30], interval=1))
    ax.tick_params(axis='both', which='major', labelsize=16)                      
    # Y axis
    if token == 'up':
        plt.ylim(sign*-0.5, sign*10)
    else:
        plt.ylim(-5, 0.1)
    plt.ylabel('Leverage After Rebalance', fontsize=axis_titles)
    
    loc = 'upper left' if token == 'up' else 'lower left'
    ax.legend(frameon=False, fontsize=legends, loc=loc)
    fig = fig.get_figure()
    fig.savefig(f'{plot_path}/{model2_path}/model2{full}_{currency.upper()}{token.upper()}_lambdast', bbox_inches='tight')
    plt.show()
'''
#%% Plot Lambda star (both tokens)


fig, ax = plt.subplots(figsize=(12,8), dpi=300)
x = sushi_merged3.timestamp

for token in ['up', 'down']:
    sign = 1 if token == 'up' else -1
    for combo in simulations:
        # only generate lables once:
        if token == 'up':
            label = label_options[combo]
        else:
            label = None
        # set colour and linestyle based on orderbook
        if combo == 'no_book':
            y = sign * simulations[combo][f'lambdast_{token}']
            ax.plot(x, y, color='k', ls='--', label=label)
        elif combo == 'no_balance':
            y = sign * simulations[combo][f'lambdast_{token}']
            ax.plot(x, y, color='k', ls='--', alpha=0.5, label=label)
        else:
            color = 'r' if 's' in combo else 'g' # decide colour based on depth
            alpha = 0.3 if 'n' in combo else 1.0 # same as above but lighter if 't' in combo else darker
            ls = '-' if 't' in combo else '--' # decide line style based on width
            y = sign * simulations[combo][f'lambdast_{token}']
            ax.plot(x, y, color=color, ls=ls, alpha=alpha, label=label)
    
    # Leverage boundaries
    plt.axhline(y=sign*4, color='k', linestyle='--', linewidth=1) # upper
    plt.axhline(y=sign*1.25, color='k', linestyle='--', linewidth=1) # lower

plt.axhline(y=0, color='k', linestyle='-', linewidth=1)

# X axis
plt.xlabel('Timestamp', fontsize=axis_titles)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(MinuteLocator(byminute=[0, 30], interval=1))
ax.tick_params(axis='both', which='major', labelsize=16)                      
# Y axis
plt.ylim(-5, 10)
plt.ylabel('Leverage After Rebalance', fontsize=axis_titles)

loc = 'upper left'
ax.legend(frameon=False, loc=loc, fontsize=legends)
fig = fig.get_figure()

# Add titles to the y-axis
ax.annotate('DOWN token',(0,4), (1,4))#, fontsize=12, ha='center', rotation=90)
ax.annotate('UP token', (0,-2), (1,-2))#, fontsize=12, ha='center', rotation=90)

fig.savefig(f'{plot_path}/{model2_path}/model2{full}_{currency.upper()}_lambdast', bbox_inches='tight')
plt.show()
