import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.dates import MinuteLocator
from pathlib import Path


currency = "sushi"
frequency = "1s"

# Create output folder for plots
output_folder = Path(f"figures/{currency}_{frequency}")
output_folder.mkdir(parents=True, exist_ok=True)
print(f"Plots will be saved to: {output_folder}")

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


# Load simulation results
simulations = {}
results_path = Path("dissertation_data/results")
if results_path.exists():
    for csv_file in sorted(results_path.glob(f"{currency}_*_simulation*.csv")):
        sim_name = csv_file.stem.replace(f"{currency}_1m_simulation_", "").replace(f"{currency}_simulation_", "")
        sim_data = pd.read_csv(csv_file)
        simulations[sim_name] = sim_data
        print(f"Loaded simulation: {sim_name}")

print(f"Total simulations loaded: {len(simulations)}")


# Plot font sizes
fsize_axis_titles = 18
fsize_legends = 12

# Label options for simulations
label_options = {
    'no_orderbook': 'No Orderbook (Baseline)',
    'orderbook_1': 'Orderbook 1',
    'orderbook_2': 'Orderbook 2',
    'orderbook_3': 'Orderbook 3',
    'orderbook_4': 'Orderbook 4',
    'orderbook_5': 'Orderbook 5',
    'orderbook_6': 'Orderbook 6',
    'orderbook_7': 'Orderbook 7',
    'orderbook_8': 'Orderbook 8',
}


# Plotting market price with simulations
fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
x = plot_df['timestamp'].values
market_price = plot_df['price'].values

# Plot simulated prices
for sim_name, sim_data in simulations.items():
    if len(sim_data) == len(plot_df):
        label = label_options.get(sim_name, sim_name)
        y = market_price * sim_data['price_multiplier'].values
        ax.plot(x, y, alpha=0.6, label=f'{label}')

# Plot actual market price last so it's on top
ax.plot(x, market_price, color='black', linewidth=2.5, label='Actual Market Price', zorder=5)

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(MinuteLocator(byminute=[0, 15, 30, 45], interval=1))
plt.xlabel('Timestamp', fontsize=fsize_axis_titles)
ax.tick_params(axis='x', which='major', labelsize=14, rotation=45)

# Format y-axis
plt.ylabel(f'{currency.upper()} Price (USDT)', fontsize=fsize_axis_titles)
ax.tick_params(axis='y', which='major', labelsize=14)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0)

# Legend and styling
ax.legend(loc='best', frameon=True, fontsize=fsize_legends, shadow=True)
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(output_folder / 'market_price.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_folder / 'market_price.png'}")
plt.close()


# Plotting UP and DOWN token NAVs
for token in ['up', 'down']:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6), dpi=300)
    x = plot_df.timestamp
    market_nav = plot_df[f'{token}_price'] * plot_df[f'nTokens{token.upper()}']

    # Plot simulated NAVs
    for sim_name, sim_data in simulations.items():
        if len(sim_data) == len(plot_df):
            label = label_options.get(sim_name, sim_name)
            y = sim_data[f'v_{token}'].values
            ax.plot(x, y, alpha=0.6, label=f'{label} (Simulated)')

    # Plot actual NAV last so it's on top
    ax.plot(x, market_nav, color='k', linewidth=2.5, label="Actual NAV", zorder=5)

    # format the x-axis
    plt.xlabel('Timestamp', fontsize=fsize_axis_titles)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(MinuteLocator(byminute=[0, 30], interval=1))
    ax.tick_params(axis='both', which='major', labelsize=16)

    # format the y-axis
    plt.ylabel(f'SUSHI{token.upper()} NAV (USDT)', fontsize=fsize_axis_titles)
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: format(int(x/1000000), ',') + 'M'))
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left', frameon=False, fontsize=fsize_legends)

    plt.tight_layout()
    plt.savefig(output_folder / f'{token}_nav.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_folder / f'{token}_nav.png'}")
    plt.close()