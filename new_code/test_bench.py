# Imports
import pandas as pd
import plotly.graph_objects as go

from new_code.price_spillover_simulations import run_simulation

# Parameters
currency = "btc"
frequency = "15s"

lambda_target = None  # None = boundary rebalancing, float = target rebalancing
lambda_upper = 4  # upper boundary
lambda_lower = 1.25  # lower boundary

# Orderbooks
# width and spread in %
orderbook_formula = "curved"
k = 0.5

orderbooks = {
    "no_orderbook" : None,
    "Deep Narrow Tight"     : {"depth":50_000_000, "width":1,  "spread":0.005},
    "Deep Narrow Broad"     : {"depth":50_000_000, "width":1,  "spread":0.05},
    "Deep Wide Tight"       : {"depth":50_000_000, "width":10, "spread":0.005},
    "Shallow Narrow Tight"  : {"depth":5_000_000,  "width":1,  "spread":0.005},
    "Asymmetrical Depth"    : {"depth":(30_000_000, 50_000_000),  "width":1,  "spread":0.005},
}

# Display options
show_hover_info = True
show_markers = False



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
            lambda_upper=lambda_upper,
            lambda_lower=lambda_lower,
            orderbook=orderbook,
            orderbook_formula=orderbook_formula,
            k=k,
            prepared_data=binance_data
        )
        results[orderbook_name] = result

    # Plot all simulations together with Plotly
    print("Plotting the data")
    fig = go.Figure()
    x = binance_data['timestamp'].values
    market_price = binance_data['price'].values

    # Plot each simulation
    dash_styles = ['solid', '6 3', '6 3 1 3', 'solid', '6 3', '6 3 1 3', 'solid', '6 3', '6 3 1 3', 'solid']
    markers = ['circle', 'square', 'triangle-up', 'diamond', 'triangle-down', 'pentagon', 'hexagon', 'cross', 'x', 'star']
    marker_offsets = [0, 3, 6, 1, 4, 7, 2, 5, 8, 0]

    for (orderbook_name, result), dash, marker, offset in zip(results.items(), dash_styles, markers, marker_offsets):
        simulated_price = market_price * result['price_multiplier'].values
        marker_indices = list(range(offset, len(x), 10))

        trace_mode = 'lines+markers' if show_markers else 'lines'
        fig.add_trace(go.Scatter(
            x=x, y=simulated_price,
            mode=trace_mode,
            name=orderbook_name,
            line={"dash": dash, "width": 2},
            marker={"size": 6, "symbol": marker, "line": {"width": 1, "color": 'white'}} if show_markers else None,
            showlegend=True,
            opacity=1,
            hovertemplate='<b>%{fullData.name}</b><br>Time: %{x|%H:%M:%S}<br>Price: $%{y:.2f}<extra></extra>' if show_hover_info else None,
            hoverinfo='skip' if not show_hover_info else None,
        ))

    # Plot actual market price on top
    fig.add_trace(go.Scatter(
        x=x, y=market_price,
        mode='lines',
        name='Actual Market Price',
        line={"color": 'black', "width": 2},
        hovertemplate='<b>Actual Market Price</b><br>Time: %{x|%H:%M:%S}<br>Price: $%{y:.2f}<extra></extra>' if show_hover_info else None,
        hoverinfo='skip' if not show_hover_info else None,
    ))

    # Format title and labels
    title_freq = frequency.replace('min', 'min ')
    fig.update_layout(
        title=f'{currency.upper()} {title_freq} Simulations Comparison',
        title_x=0.5,
        title_xanchor='center',
        title_font_size=28,
        xaxis_title='Timestamp',
        yaxis_title=f'{currency.upper()} Price (USDT)',
        hovermode='closest',
        template='plotly_white',
        height=800,
        font={"size": 12, "color": 'black'},
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend={"x": 0.02, "y": 0.05, "bgcolor": 'rgba(255, 255, 255, 0.9)', "bordercolor": 'black', "borderwidth": 1, "xanchor": 'left', "yanchor": 'bottom', "font": {"color": 'black', "size": 12}},
        title_font_color='black'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_font_color='black', tickfont_color='black')

    # Set y-axis range: 0 to 1.2 * max market price
    max_price = market_price.max()
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_font_color='black', tickfont_color='black', range=[0, 1.2 * max_price])

    fig.show()