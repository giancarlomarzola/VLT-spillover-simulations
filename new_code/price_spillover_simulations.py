import pandas as pd
import numpy as np

currency = "btc"
frequency = "1m"

# Parameters (same for up and down)
lambda_target = 3  # Only used when rebalance = 'target'
lambda_up = 4  # upper boundary
lambda_down = 1.25  # lower boundary
# lambda_0     = 3.5 # Leverage at start
rebalance = "boundary"  # either 'boundary' or 'target'


# import dataframe
filename = f"{currency}_data" + (f"_{frequency}" if frequency else "")
df = pd.read_csv(f"dissertation_data/token_dataframes/{filename}.csv")


# Ensure timestamp is datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Filter to analysis period
analysis_start = pd.Timestamp("2021-05-19 12:00:00")
analysis_end = pd.Timestamp("2021-05-19 14:00:00")

df = df[df["timestamp"].between(analysis_start, analysis_end)].reset_index(drop=True)


# Get initial 
underlying_price = np.array(df["price"].values)

# UP parameters
up_investment = df["nTokensUP"].iloc[0] * df["up_price"].iloc[0]
up_lambda_0 = df["leverageUP"].iloc[0]
# DOWN Parameters
down_investment = df["nTokensDOWN"].iloc[0] * df["down_price"].iloc[0]
down_lambda_0 = df["leverageDOWN"].iloc[0]
