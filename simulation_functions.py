# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 18:21:47 2023

@author: gm399
"""

import os
import pandas as pd
import numpy as np

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



def token_characteristics(underlying_price, lambda_target, lambda_up, lambda_down,
                          up_investment, up_lambda_0, down_investment, down_lambda_0,
                          rebalance, orderbook=None):
    # Begin Simulation
    price_array = underlying_price.copy() #prevent price array from being overwritten
    nvars = 7 # v, x, x*, lambda, lambda*, delta, price_multiplier
    out = np.empty((len(price_array), nvars*2+4))
    omegas = np.array([1,-1])

    # initialize the variables at time t=0

    # Allows to get columns by using var names
    '''
    v                 = i*nvars+0
    x                 = i*nvars+1
    x_st              = i*nvars+2
    lam               = i*nvars+3
    lam_st            = i*nvars+4
    target_delta      = i*nvars+5
    actual_delta      = i*nvars+6
    
    delta_target = 12 # necessary total rebalance
    delta_actual      = 13 # actual rebalance that can be carried out, given orderbook
    price_mult        = 14 # cumulative price effect to date
    price_effect      = 15 # effect only at time t
    '''

    # Initialize up vars (cols 0 - 5)
    omega = 1
    i = 0
    out[0, i*nvars+0] = up_investment                                # v
    out[0, i*nvars+1] = omega * up_investment * up_lambda_0          # x
    out[0, i*nvars+2] = omega * up_investment * up_lambda_0          # x_st
    out[0, i*nvars+3] = omega * out[0,i*nvars+1] / out[0,i*nvars+0]  # lambda
    out[0, i*nvars+4] = out[0,i*nvars+3]                             # lambda_st
    out[0, i*nvars+5] = 0                                            # target_delta
    out[0, i*nvars+6] = 0                                            # actual_delta

    # Initialiaze down vars (cols 6 - 11)
    omega = -1
    i = 1
    out[0, i*nvars+0] = down_investment                              # v
    out[0, i*nvars+1] = omega * down_investment * down_lambda_0      # x
    out[0, i*nvars+2] = omega * down_investment * down_lambda_0      # x_st
    out[0, i*nvars+3] = omega * out[0,i*nvars+1] / out[0,i*nvars+0]  # lambda
    out[0, i*nvars+4] = out[0,i*nvars+3]                             # lambda_st
    out[0, i*nvars+5] = 0                                            # target_delta
    out[0, i*nvars+6] = 0                                            # actual_delta

    # Other vars (cols 12 - 16)
    out[0, 2*nvars+0] = out[0,0*nvars+5] + out[0,1*nvars+5]   # target_delta_total
    out[0, 2*nvars+1] = out[0,2*nvars+0]                      # actual_delta_total
    out[0, 2*nvars+2] = 1                                     # price_multiplier
    out[0, 2*nvars+3] = 0                                     # price effect from orderbook
    
    # update variables over time
    for t in range(1,len(out)):
        
        #apply price multiplier
        if orderbook is not None:
            price_array[t] = price_array[t] * out[t-1,2*nvars+2]
        
        # get return for the current time step
        ret = price_array[t]/price_array[t-1]-1
        
        # For up and down token:
        for i in range(len(omegas)):
            
            #update v, x, and lambda
            omega = omegas[i]
            out[t,i*nvars+0] = out[t-1,i*nvars+0] * (1 + omega * out[t-1,i*nvars+4] * ret) # v
            if out[t,i*nvars+0] <= 0: # If v reaches 0 or below
            # If this works: could simply set rows for that token = 0
                out[t,0:i*nvars+nvars] = 0 
                continue
                
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
        
        # After getting up and down token - get combined effect 
        
        # target total (combined) delta in $
        out[t,2*nvars+0] = out[t,0*nvars+5] + out[t,1*nvars+5]
            
        # Orderbook effect of the rebalances:
        if (orderbook is not None) and (out[t,2*nvars+0] != 0):
            # get combined delta in tokens
            delta_tot_tokens = out[t,2*nvars+0] / price_array[t]
            
            # Use delta and orderbook to find price effect
            if delta_tot_tokens > 0:
                side = 'ask'
                sign = 1
            else:
                side = 'bid'
                sign = -1
            
            nearest_depth = orderbook[orderbook[f"cumulative_{side}"] <= (sign*delta_tot_tokens)][f'cumulative_{side}'].max()
            price_difference = orderbook.loc[orderbook[f'cumulative_{side}'] == nearest_depth, 'price'].min() 
            price_difference = sign * price_difference / 100 # as decimal
            
            # Update price difference and multiplier
            out[t,2*nvars+2] = out[t-1,2*nvars+2] * (1+price_difference) # update price multiplier
            out[t,2*nvars+3] = price_difference
            
            # If max depth is exceeded and delta is capped:
            if abs(delta_tot_tokens) > orderbook[f'cumulative_{side}'].max():
                # Cap total delta
                out[t,2*nvars+1] = sign * nearest_depth * price_array[t] # actual delta in $
                # Recalculate UP and DOWN values
                for i in range(len(omegas)):
                    omega = omegas[i]
                    # Don't recalculate if v is 0
                    if out[t,i*nvars+0] <= 0:
                        out[t,0:i*nvars+nvars] = 0 
                        continue
                    # Actual delta:
                    proportion = out[t,i*nvars+5] / out[t,2*nvars+0] # target delta_token / target_total_delta
                    out[t,i*nvars+6] = out[t,2*nvars+1]*proportion # Actual delta = target_delta * proportion
                    # New x_st
                    out[t, i*nvars+2] = out[t, i*nvars+1] + out[t,i*nvars+6]# x + delta
                    # New lambda_st (does not reach target lambda)
                    out[t,i*nvars+4] = omega * out[t, i*nvars+2] / out[t,i*nvars+0]
                    
                                
            else: # if delta does not exceed depth
                out[t,2*nvars+1] = out[t,2*nvars+0] # actual delta total == target delta (in $)
                # up and down actual deltas == up and down target deltas
                for i in range(len(omegas)):
                    out[t,i*nvars+6] = out[t,i*nvars+5]
                    
        else: # no rebalance
            out[t,2*nvars+1] = out[t,2*nvars+0] # actual delta total = target delta = 0
            out[t,2*nvars+2] = out[t-1,2*nvars+2] # price multiplier stays same
            out[t,2*nvars+3] = 0 # price effect is 0
            # up and down actual deltas == up and down target deltas = 0
            for i in range(len(omegas)):
                out[t,i*nvars+6] = out[t,i*nvars+5]
            
    # Make into dataframe            
    tk = pd.DataFrame(out, columns=["v_up", "x_up", "xst_up", 
                                    "lambda_up", "lambdast_up", 
                                    "target_delta_up", "actual_delta_up",
                                    "v_down", "x_down", "xst_down", 
                                    "lambda_down", "lambdast_down", 
                                    "target_delta_down", "actual_delta_down",
                                    "target_total_delta", "actual_total_delta", 
                                    "price_multiplier", "orderbook_effect"])
    
    return(tk)