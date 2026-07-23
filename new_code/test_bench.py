# Imports
from new_code.price_spillover_simulations import run_simulation
import plotly.graph_objects as go
import pandas as pd

# Parameters
currency = "btc"
frequency = "15s"

lambda_target = None  # None = boundary rebalancing, float = target rebalancing
lambda_up = 4  # upper boundary
lambda_down = 1.25  # lower boundary

# Orderbooks
# width and spread in %
orderbooks = {
    "no_orderbook" : None,
    "Deep Narrow Tight"     : {"depth":50_000_000, "width":1,  "spread":0.005},
    "Deep Narrow Broad"     : {"depth":50_000_000, "width":1,  "spread":0.05},
    "Deep Wide Tight"       : {"depth":50_000_000, "width":10, "spread":0.005},
    "Shallow Narrow Tight"  : {"depth":5_000_000,  "width":1,  "spread":0.005},
    "Asymmetrical Depth"    : {"depth":(30_000_000, 50_000_000),  "width":1,  "spread":0.005},
}

# Display options
show_hover_info = False



if __name__ == "__main__":
    # Load pre-processed data
    try:
        print(f"Loading {currency.upper()} at {frequency} frequency...")
        binance_data = pd.read_parquet(f"dissertation_data/token_dataframes/{currency}_{frequency}_processed.parquet")
        binance_data["timestamp"] = pd.to_datetime(binance_data["timestamp"], utc=True)
        print(f"Loaded {len(binance_data)} rows")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Pre-processed data not found for {currency} at {frequency} frequency.\n"
            f"Run data_processing.py to prepare the data."
        )

    # Simulation
    results = {}

    for orderbook_name, orderbook in orderbooks.items():
        print(f"Running simulation for {orderbook_name}")
        result = run_simulation(
            lambda_target=lambda_target,
            lambda_up=lambda_up,
            lambda_down=lambda_down,
            orderbook=orderbook,
            data=binance_data
        )
        results[orderbook_name] = result

    # Plot all simulations together with Plotly
    print("Plotting the data")
    fig = go.Figure()
    x = binance_data['timestamp'].values
    market_price = binance_data['price'].values

    # Plot each simulation with staggered markers to avoid perfect overlap
    line_styles = ['-', '--', '-.', '-', '--', '-.', '-', '--', '-.', '-']
    dash_styles = ['solid', 'dash', 'dashdot', 'solid', 'dash', 'dashdot', 'solid', 'dash', 'dashdot', 'solid']
    markers = ['circle', 'square', 'triangle-up', 'diamond', 'triangle-down', 'pentagon', 'hexagon', 'cross', 'x', 'star']
    marker_offsets = [0, 3, 6, 1, 4, 7, 2, 5, 8, 0]

    for (orderbook_name, result), dash, marker, offset in zip(results.items(), dash_styles, markers, marker_offsets):
        simulated_price = market_price * result['price_multiplier'].values
        marker_indices = list(range(offset, len(x), 10))
        fig.add_trace(go.Scatter(
            x=x, y=simulated_price,
            mode='lines+markers',
            name=orderbook_name,
            line=dict(dash=dash, width=2),
            marker=dict(size=6, symbol=marker, line=dict(width=1, color='white')),
            showlegend=True,
            opacity=1,
            hovertemplate='<b>%{fullData.name}</b><br>Time: %{x|%H:%M:%S}<br>Price: $%{y:.2f}<extra></extra>' if show_hover_info else None,
            hoverinfo='skip' if not show_hover_info else None,
            visible=True
        ))

        # Add markers at specific positions
        marker_x = [x[i] for i in marker_indices if i < len(x)]
        marker_y = [simulated_price[i] for i in marker_indices if i < len(x)]
        if marker_x:
            fig.add_trace(go.Scatter(
                x=marker_x, y=marker_y,
                mode='markers',
                name=orderbook_name,
                marker=dict(size=8, symbol=marker, line=dict(width=1, color='white')),
                showlegend=False,
                opacity=0.6,
                hoverinfo='skip' if True else None,
                visible=True
            ))

    # Plot actual market price on top
    fig.add_trace(go.Scatter(
        x=x, y=market_price,
        mode='lines',
        name='Actual Market Price',
        line=dict(color='black', width=1.5),
        hovertemplate='<b>Actual Market Price</b><br>Time: %{x|%H:%M:%S}<br>Price: $%{y:.2f}<extra></extra>' if show_hover_info else None,
        hoverinfo='skip' if not show_hover_info else None,
        visible=True
    ))

    # Format title and labels
    title_freq = frequency.replace('min', 'min ').upper()
    fig.update_layout(
        title=f'{currency.upper()} {title_freq} Simulations Comparison',
        xaxis_title='Timestamp',
        yaxis_title=f'{currency.upper()} Price (USDT)',
        hovermode='x unified' if show_hover_info else False,
        template='plotly_white',
        width=1400,
        height=700,
        font=dict(size=12),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)', bordercolor='black', borderwidth=1)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    fig.show()