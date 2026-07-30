import numpy as np
import pandas as pd
from IPython.display import display

currency = 'btc'
frequency = '50ms'

df = pd.read_parquet(f"dissertation_data/token_dataframes/{currency}_{frequency}_processed.parquet")

df['pct_return'] = df['price'].pct_change()

df['log_return'] = np.log(df['price'] / df['price'].shift(1))

# Log returns
print("Summary statistics for log returns (in basis points)")
log_returns_bp = (df['log_return']*100).copy()
display(log_returns_bp.describe().apply("{0:,.5f}".format))
print(f"1st quantile:\t {log_returns_bp.quantile(0.01)}")
print(f"99th quantile:\t  {log_returns_bp.quantile(0.99)}")


# Percentage change
print("Summary statistics for percent returns (in basis points)")
pct_returns_bp = (df['pct_return']*100).copy()
display(pct_returns_bp.describe().apply("{0:,.5f}".format))
print(f" 1% quantile:\t {pct_returns_bp.quantile(0.01)}")
print(f"99% quantile:\t  {pct_returns_bp.quantile(0.99)}")