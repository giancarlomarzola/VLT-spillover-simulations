import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MinuteLocator


currency = "sushi"
frequency = "1m"

# import dataframe
filename = f"{currency}_data" + (f"_{frequency}" if frequency else "")
plot_df = pd.read_csv(f"dissertation_data/token_dataframes/{filename}.csv")


# Ensure timestamp is datetime format
plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'])

# Filter to analysis period
analysis_start = pd.Timestamp('2021-05-19 12:00:00')
analysis_end = pd.Timestamp('2021-05-19 14:00:00')

plot_df = plot_df[plot_df['timestamp'].between(analysis_start, analysis_end)]

if len(plot_df) == 0:
    print(f"Warning: No data found between {analysis_start} and {analysis_end}")
else:
    print(f"Loaded {len(plot_df)} records from {plot_df['timestamp'].min()} to {plot_df['timestamp'].max()}")


# Plot font sizes
fsize_axis_titles = 18
fsize_legends = 14

# Plotting actual market price
fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
x = plot_df['timestamp'].values
market_price = plot_df['sushi_price'].values

ax.plot(x, market_price, color='black', linewidth=2.5, label='Actual Market Price', zorder=5)

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(MinuteLocator(byminute=[0, 15, 30, 45], interval=1))
plt.xlabel('Timestamp', fontsize=fsize_axis_titles)
ax.tick_params(axis='x', which='major', labelsize=14, rotation=45)

# Format y-axis
plt.ylabel('SUSHI Price (USDT)', fontsize=fsize_axis_titles)
ax.tick_params(axis='y', which='major', labelsize=14)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0)

# Legend and styling
ax.legend(loc='best', frameon=True, fontsize=fsize_legends, shadow=True)
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.show()


# Plotting UP and DOWN token NAVs
for token in ['up', 'down']:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), dpi=300)
    x = plot_df.timestamp
    market_nav = plot_df[f'{currency}_{token}_price']*plot_df[f'nTokens{token}']
    label0 = "Actual NAV"

    ax.plot(x, market_nav, color='k', label=label0)
    
    # format the x-axis
    plt.xlabel('Timestamp', fontsize=fsize_axis_titles)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(MinuteLocator(byminute=[0, 30], interval=1))
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # format the y-axis
    plt.ylabel(f'SUSHI{token} NAV (Million USDT)', fontsize=fsize_axis_titles)
    #plt.ylim(-1_000_000,41_000_000)
    #ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x/1000000), ',')))
    
    # Legend
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left', frameon=False, fontsize=fsize_legends)
    
    plt.show()