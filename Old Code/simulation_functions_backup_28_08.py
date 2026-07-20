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
    nvars = 6 # v, x, x*, lambda, lambda*, delta, price_multiplier
    out = np.zeros((len(price_array), nvars*2+3))
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
    delta_theoretical = 12
    delta_actual      = 13    
    price_mult        = 14
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
    out[0, 14] = 0                                                   # price effect from orderbook
    
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
            if out[t,i*nvars+0] <= 0: # If NAV reaches 0 or below
            # If this works: could simply set rows 0:12 = 0
                out[t,i*nvars+0] = 0 # set NAV to 0
                out[t,i*nvars+1] = 0 # Set x (notional position) to 0
                out[t,i*nvars+3] = 0 # set Lambda to 0
                out[t, 13] = out[t-1, 13] # Keep price multiplier equal
                out[t, 14] = 0 # Set price effect to 0
                out[t, i*nvars+2] = 0 # et x_st to 0
                out[t,i*nvars+4] = 0 # Set lambda_st to 0
                out[t,i*nvars+5] = 0 # set delta to 0
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
            
            # Total (combined) delta
            out[t, 12] = out[t,0*nvars+5] + out[t,1*nvars+5]
            
            # Orderbook effect of the rebalances:
            if (orderbook is not None) and (out[t, 12] != 0):
                # get combined delta
                delta_tot_tokens = out[t, 12] / price_array[t]
                
                # Use delta and orderbook to find price effect
                
                # With positive delta (use up ask side), no change necessary
                if delta_tot_tokens > 0:
                    nearest_depth = orderbook[orderbook['cumulative_ask'] <= delta_tot_tokens]['cumulative_ask'].max()
                    price_difference = orderbook.loc[orderbook['cumulative_ask'] == nearest_depth, 'price'].iloc[0]
                    out[t,13] = out[t-1,13] * (1+price_difference/100) # update price multiplier
                    out[t, 14] = price_difference/100

                # With negative delta (use up bid side) need to make positive, look up, then convert to negative again
                elif delta_tot_tokens < 0:
                    nearest_depth = orderbook[orderbook['cumulative_bid'] <= -delta_tot_tokens]['cumulative_bid'].max()
                    price_difference = orderbook.loc[orderbook['cumulative_bid'] == nearest_depth, 'price'].iloc[0]
                    out[t,13] = out[t-1,13] * (1-price_difference/100)
                    out[t, 14] = -price_difference/100
            
            else:
                # Multiplier stays the same
                out[t, 13] = out[t-1, 13]
                out[t, 14] = 0
                
        tk = pd.DataFrame(out, columns=["v_up", "x_up", "xst_up", "lambda_up", "lambdast_up", "delta_up",  
                                        "v_down", "x_down", "xst_down", "lambda_down", "lambdast_down", "delta_down",
                                        "combined_delta", "price_multiplier", "orderbook_effect"])
    
    return(tk)