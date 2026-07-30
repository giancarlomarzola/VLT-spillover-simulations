import random

import pandas as pd
import plotly.graph_objects as go

from new_code.price_spillover_simulations import run_simulation


def plot_results(results_dict, lambda_upper=4.0, lambda_lower=1.25):
    """Plot prices and leverage from simulation results dict.

    Args:
        results_dict: Dict of result dataframes
        lambda_upper: Upper leverage boundary for visualization
        lambda_lower: Lower leverage boundary for visualization
    """
    trace_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    # Create price plot
    price_fig = go.Figure()
    first_df = next(iter(results_dict.values()))
    x_axis = list(range(len(first_df)))

    # Plot raw_price from first dataframe
    price_fig.add_trace(go.Scatter(
        x=x_axis, y=first_df['raw_price'],
        mode='lines',
        name='Actual Market Price',
        line={"color": 'black', "width": 2},
    ))

    # Plot simulated_price for all dataframes
    for idx, (name, df) in enumerate(results_dict.items()):
        color = trace_colors[idx % len(trace_colors)]
        price_fig.add_trace(go.Scatter(
            x=x_axis, y=df['simulated_price'],
            mode='lines',
            name=name,
            line={"color": color, "width": 2},
        ))

    price_fig.update_layout(
        title='Price Simulations Comparison',
        title_x=0.5,
        title_xanchor='center',
        title_font_size=28,
        xaxis_title='Time',
        yaxis_title='Price',
        hovermode='closest',
        template='plotly_white',
        height=800,
        font={"size": 12, "color": 'black'},
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend={"x": 0.02, "y": 0.02, "bgcolor": 'rgba(255, 255, 255, 0.9)', "bordercolor": 'black', "borderwidth": 1, "xanchor": 'left', "yanchor": 'bottom'},
        margin={"l": 80, "r": 80, "t": 100, "b": 80},
    )
    price_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    price_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Create leverage plot
    leverage_fig = go.Figure()

    for idx, (name, df) in enumerate(results_dict.items()):
        color = trace_colors[idx % len(trace_colors)]
        # Plot UP leverage
        leverage_fig.add_trace(go.Scatter(
            x=x_axis, y=df['lambdast_up'],
            mode='lines',
            name=f'{name} UP',
            line={"color": color, "width": 2},
        ))
        # Plot DOWN leverage (negated for display)
        if 'lambdast_down' in df.columns:
            leverage_fig.add_trace(go.Scatter(
                x=x_axis, y=-df['lambdast_down'],
                mode='lines',
                name=f'{name} DOWN',
                line={"color": color, "width": 2, "dash": '6 3'},
            ))

    leverage_fig.update_layout(
        title='Leverage (λ) Evolution',
        title_x=0.5,
        title_xanchor='center',
        title_font_size=28,
        xaxis_title='Time',
        yaxis_title='Leverage (λ)',
        hovermode='closest',
        template='plotly_white',
        height=600,
        font={"size": 12, "color": 'black'},
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend={"x": 0.98, "y": 0.02, "bgcolor": 'rgba(255, 255, 255, 0.9)', "bordercolor": 'black', "borderwidth": 1, "xanchor": 'right', "yanchor": 'bottom'},
        margin={"l": 80, "r": 80, "t": 100, "b": 80},
    )
    leverage_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    leverage_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=True, zerolinecolor='gray', zerolinewidth=1)

    # Add lambda bounds
    for threshold, label in [(lambda_upper, 'λ_upper'), (lambda_lower, 'λ_lower')]:
        for sign in (1, -1):
            leverage_fig.add_hline(
                y=sign * threshold,
                line={"color": 'gray', "width": 1, "dash": 'dot'},
                annotation_text=f'{label} = {sign * threshold:g}',
                annotation_position='top left',
                annotation_font_color='gray',
                annotation_font_size=11,
            )

    return price_fig, leverage_fig


# Random price series - to be substituted later
def generate_price_series(start_price=100, steps=100, volatility=0.001):
    prices = [start_price]
    for _ in range(steps - 1):
        ret = random.uniform(-volatility, volatility)
        prices.append(round(prices[-1] * (1+ret), 2))
    return prices


price_series = generate_price_series(50_000, 144_000)


lambda_target = None  # None = boundary rebalancing, float = target rebalancing
lambda_upper = 4  # upper boundary
lambda_lower = 1.25  # lower boundary

start_nav_up = 30_000_000
start_exposure_up = 90_000_000

start_nav_down = 20_000_000
start_exposure_down = -60_000_000

# Orderbooks
# width and spread in %
orderbook_formula = "curved"
k = 0.5

orderbooks = {
    "No Orderbook": None,
    "Deep Narrow Tight"     : {"depth":50_000_000, "width":1,  "spread":0.005},
    "Deep Narrow Broad"     : {"depth":50_000_000, "width":1,  "spread":0.05},
}

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
        price_series=price_series,
        start_nav_up=start_nav_up,
        start_exposure_up=start_exposure_up,
        start_nav_down=start_nav_down,
        start_exposure_down=start_exposure_down,
    )
    results[orderbook_name] = result

price_fig, leverage_fig = plot_results(results, lambda_upper=lambda_upper, lambda_lower=lambda_lower)
price_fig.show()
leverage_fig.show()