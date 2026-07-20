# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 21:49:46 2023

@author: gm399
"""

import numpy as np
import pandas as pd
from orderbook import make_orderbook

filepath = "D:/dissertation-data"

sushi_merged3 = pd.read_csv(f'{filepath}/sushi_merged3.csv')

# Parameters (same for up and down)
lambda_target = 3 # Only used when rebalance = 'target'
lambda_up     = 4 # upper boundary
lambda_down   = 1.25 # lower boundary
#lambda_0      = 3.5 # Leverage at start
rebalance     = "boundary" # either 'boundary' or 'target'

underlying_price         = np.array(sushi_merged3["sushi_price"].values)

# UP parameters
up_investment = 10_000
up_lambda_0    = sushi_merged3["leverageUP"].iloc[0]

# DOWN Parameters
down_investment = 10_000#125_716_067.04 # Number of SUSHIDOWN issued on 19.05.2021; number kept changing
down_lambda_0    = sushi_merged3["leverageDOWN"].iloc[0]

#Orderbook

# To mimic ETH on 20.08:
beta, alpha          = 0.00005, 0.00005 
worst_bid, worst_ask = 0.6, 0.6 
bid_depth, ask_depth = 19_000, 17_000 
max_deviation        = 0.7    
tick_size            = 0.00005

orderbook = make_orderbook(alpha, beta, worst_ask, worst_bid, ask_depth, bid_depth, 
                   max_deviation, tick_size)

#--------------------------------------------------------------------------

# Begin Simulation
price_array = underlying_price.copy() #prevent price array from being overwritten
nvars = 6 # v, x, x*, lambda, lambda*, delta, price_multiplier
out = np.zeros((len(price_array), nvars*2+2))
omegas = np.array([1,-1])

# initialize the variables at time t=0

# Allows to get columns by using var names
'''
v          = i*nvars+0
x          = i*nvars+1
x_st       = i*nvars+2
lam        = i*nvars+3
lam_st     = i*nvars+4
delta      = i*nvars+5
delta_tot  = 12
price_mult = 13
'''

# Initialize up vars
omega = 1
i = 0
out[0, i*nvars+0] = up_investment                                # v
out[0, i*nvars+1] = omega * up_investment * up_lambda_0          # x
out[0, i*nvars+2] = omega * up_investment * up_lambda_0          # x_st
out[0, i*nvars+3] = omega * out[0,i*nvars+1] / out[0,i*nvars+0]  # lambda
out[0, i*nvars+4] = out[0,i*nvars+3]                             # lambda_st
out[0, i*nvars+5] = 0                                            # delta

# Initialiaze down vars
omega = -1
i = 1
out[0, i*nvars+0] = down_investment                              # v
out[0, i*nvars+1] = omega * down_investment * down_lambda_0      # x
out[0, i*nvars+2] = omega * down_investment * down_lambda_0      # x_st
out[0, i*nvars+3] = omega * out[0,i*nvars+1] / out[0,i*nvars+0]  # lambda
out[0, i*nvars+4] = out[0,i*nvars+3]                             # lambda_st
out[0, i*nvars+5] = 0                                            # delta

# Other vars
out[0, 12] = out[0,0*nvars+5] + out[0,1*nvars+5]                 # delta_total
out[0, 13] = 1                                                   # price_multiplier

# update variables over time
for t in range(1,len(out)):
    
    #apply price multiplier
    if orderbook is not None:
        price_array[t] = price_array[t] * out[t-1, 13]
    
    # get return for the current time step
    ret = price_array[t]/price_array[t-1]-1
    
    # For up and down token:
    for i in range(len(omegas)):
        
        #update v, x, and lambda
        omega = omegas[i]
        out[t,i*nvars+0] = out[t-1,i*nvars+0] * (1 + omega * out[t-1,i*nvars+4] * ret) # v
        out[t,i*nvars+1] = out[t-1,i*nvars+2] * (1 + ret) # x
        out[t,i*nvars+3] = omega * out[t,i*nvars+1] / out[t,i*nvars+0]  # lambda
    
        # update x_star
        # check whether rebalancing is required
        if lambda_down <= out[t,i*nvars+3] <= lambda_up:
            out[t, i*nvars+2] = out[t,i*nvars+1] # if no rebalance, x_star = x
        else: # rebalance back to target
            if rebalance=="boundary":
                if out[t, i*nvars+3] <= lambda_down: # rebalance from below
                    out[t, i*nvars+2] = omega*out[t,i*nvars+0]*lambda_down # x_star = omega*v*target_lambda (lower)
                elif out[t, i*nvars+3] >=lambda_up: # rebalance from above
                    out[t, i*nvars+2] = omega*out[t,i*nvars+0]*lambda_up # x_star = omega*v*target_lambda (upper)
            elif rebalance=="target":
                out[t, i*nvars+2] = omega*out[t,i*nvars+0]*lambda_target
    
        out[t,i*nvars+4] = omega * out[t, i*nvars+2] / out[t,i*nvars+0] # lambda_star
        out[t,i*nvars+5] = out[t, i*nvars+2] - out[t,i*nvars+1] # delta = x_star - x in $ - convert to sushi
        
        # Total (combined) delta
        out[t, 12] = out[t,0*nvars+5] + out[t,1*nvars+5]
        
        # Orderbook effect of the rebalances:
        if (orderbook is not None) and (out[t, 12] != 0):
            # get combined delta
            delta_tot_tokens = out[t, 12] / price_array[t]
            
            # Use delta and orderbook to find price effect
            
            # With positive delta, no change necessary
            if delta_tot_tokens > 0:
                nearest_depth = orderbook[orderbook['cumulative_bid'] <= delta_tot_tokens]['cumulative_bid'].max()
                price_difference = orderbook.loc[orderbook['cumulative_bid'] == nearest_depth, 'price'].iloc[0]
                out[t,13] = out[t-1,13] * (1+price_difference) # update price multiplier

            # With negative delta need to make positive, look up, then convert to negative again
            elif delta_tot_tokens < 0:
                nearest_depth = orderbook[orderbook['cumulative_bid'] <= -delta_tot_tokens]['cumulative_bid'].max()
                price_difference = orderbook.loc[orderbook['cumulative_bid'] == nearest_depth, 'price'].iloc[0]
                out[t,13] = out[t-1,13] * (1-price_difference)
        
        else:
            # Multiplier stays the same
            out[t, 13] = out[t-1, 13]
            
    tk = pd.DataFrame(out, columns=["v_up", "x_up", "xst_up", "lambda_up", 
                                    "lambdast_up", "delta_up",  
                                    "v_down", "x_down", "xst_down", "lambda_down", 
                                    "lambdast_down", "delta_down", 
                                    "combined_delta", "price_effect_down"])