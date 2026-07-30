"""Shared plotting utilities for simulation results."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TRACE_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def resample_data(data, resample_freq):
    """Resample time-series data to a coarser frequency by taking the last value in each period.

    Args:
        data: DataFrame with 'timestamp' column
        resample_freq: Resample frequency string (e.g., '15s', '1min') or 'raw' for no resampling

    Returns:
        Resampled DataFrame
    """
    if resample_freq == "raw":
        return data.copy()

    df = data.copy()
    df.set_index("timestamp", inplace=True)
    resampled = df.resample(resample_freq).last()
    resampled.reset_index(inplace=True)
    return resampled


def plot_results(
    results_dict,
    market_price=None,
    x_axis=None,
    lambda_upper=4.0,
    lambda_lower=1.25,
    show_hover=True,
    show_markers=False,
    leverage_timing="Before Rebalance",
    title_prefix="",
    rebalance_magnitudes=None,
    currency="",
    resample_freq=None,
):
    """Plot prices and leverage from simulation results.

    Args:
        results_dict: Dict of result dataframes
        market_price: Array of actual market prices (if None, uses raw_price from first dataframe)
        x_axis: X-axis values (if None, uses list of indices). Should be timestamps if resampling.
        lambda_upper: Upper leverage boundary for visualization
        lambda_lower: Lower leverage boundary for visualization
        show_hover: Whether to show hover info
        show_markers: Whether to show markers on price lines
        leverage_timing: "Before Rebalance" or "After Rebalance"
        title_prefix: Prefix for plot titles (e.g., "BTC 30s")
        rebalance_magnitudes: Dict with rebalance magnitudes for each orderbook (optional)
        currency: Currency symbol for axis labels (default '')
        resample_freq: Resample frequency (e.g., '30s', '1min', '15min'). None = no resampling.

    Returns:
        Tuple of (price_fig, leverage_fig)
    """
    # Extract timestamps from first result if available
    first_df = next(iter(results_dict.values()))
    if "timestamp" in first_df.columns:
        result_timestamps = first_df["timestamp"].values
        if x_axis is None:
            x_axis = result_timestamps

    # Determine which price to use as baseline (before resampling)
    if market_price is None:
        if "raw_price" in first_df.columns:
            market_price = first_df["raw_price"].values
        else:
            raise ValueError(
                "market_price must be provided or 'raw_price' must be in dataframes"
            )

    # Resample data if requested
    if resample_freq and resample_freq != "raw" and x_axis is not None and len(x_axis) > 0:
        temp_df = pd.DataFrame(
            {"timestamp": x_axis, "market_price": market_price}
        )
        temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"], utc=True)
        temp_df.set_index("timestamp", inplace=True)
        resampled = temp_df.resample(resample_freq).last()
        resampled_x_axis = resampled.index.values
        market_price = resampled["market_price"].values

        # Resample each result dataframe using its own timestamps
        resampled_results_dict = {}
        for name, df in results_dict.items():
            df_temp = df.copy()
            if "timestamp" in df_temp.columns:
                # Use timestamps already in the dataframe
                df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"], utc=True)
                df_temp.set_index("timestamp", inplace=True)
            else:
                # Fall back to x_axis if no timestamps in df
                df_temp["timestamp"] = pd.to_datetime(x_axis, utc=True)
                df_temp.set_index("timestamp", inplace=True)
            df_resampled = df_temp.resample(resample_freq).last()
            resampled_results_dict[name] = df_resampled.reset_index(drop=True)

        results_dict = resampled_results_dict
        x_axis = resampled_x_axis
    # Get data dimensions
    first_df = next(iter(results_dict.values()))
    if x_axis is None:
        x_axis = list(range(len(first_df)))

    # Calculate rebalance magnitudes if not provided
    if not rebalance_magnitudes:
        rebalance_magnitudes = {}
        for name, df in results_dict.items():
            if "actual_total_delta" in df.columns:
                rebalance_magnitudes[name] = df["actual_total_delta"].abs().values
            elif "actual_delta_up" in df.columns and "actual_delta_down" in df.columns:
                rebalance_magnitudes[name] = (
                    df["actual_delta_up"].abs() + df["actual_delta_down"].abs()
                ).values
            else:
                rebalance_magnitudes[name] = None

    # Create price plot
    price_fig = go.Figure()

    # Plot simulated prices for all results first (so market price is on top)
    for idx, (name, df) in enumerate(results_dict.items()):
        color = TRACE_COLORS[idx % len(TRACE_COLORS)]

        if "simulated_price" in df.columns:
            y_data = df["simulated_price"].values
        elif "price_multiplier" in df.columns:
            y_data = market_price * df["price_multiplier"].values
        else:
            raise ValueError(
                "Dataframe must contain 'simulated_price' or 'price_multiplier' column"
            )

        trace_mode = "lines+markers" if show_markers else "lines"
        price_fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=y_data,
                mode=trace_mode,
                name=name,
                line={"color": color, "width": 2},
                marker=(
                    {"size": 6, "line": {"width": 1, "color": "white"}}
                    if show_markers
                    else None
                ),
                hovertemplate=(
                    f"<b>{name}</b><br>Time: %{{x|%H:%M:%S}}<br>Price: $%{{y:.2f}}<extra></extra>"
                    if show_hover
                    else None
                ),
                hoverinfo="skip" if not show_hover else None,
            )
        )

        # Add rebalance events if available
        if name in rebalance_magnitudes and rebalance_magnitudes[name] is not None:
            rebalance_x, rebalance_y = [], []
            for xi, magnitude in zip(x_axis, rebalance_magnitudes[name]):
                if magnitude > 0 and not np.isnan(magnitude):
                    rebalance_x += [xi, xi, None]
                    rebalance_y += [0, magnitude, None]

            if rebalance_x:
                price_fig.add_trace(
                    go.Scatter(
                        x=rebalance_x,
                        y=rebalance_y,
                        mode="lines",
                        name=f"{name} rebalances",
                        line={"color": color, "width": 2},
                        yaxis="y2",
                        legend="legend2",
                        visible="legendonly",
                        hovertemplate=(
                            f"<b>{name} rebalance</b><br>Time: %{{x|%H:%M:%S}}<br>Size: $%{{y:,.0f}}<extra></extra>"
                            if show_hover
                            else None
                        ),
                        hoverinfo="skip" if not show_hover else None,
                    )
                )

    # Plot market price last (so it's on top)
    price_fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=market_price,
            mode="lines",
            name="Actual Market Price",
            line={"color": "black", "width": 2},
            hovertemplate=(
                "<b>Actual Market Price</b><br>Time: %{x|%H:%M:%S}<br>Price: $%{y:.2f}<extra></extra>"
                if show_hover
                else None
            ),
            hoverinfo="skip" if not show_hover else None,
        )
    )

    # Setup layout
    yaxis_title = f"{currency} Price (USDT)" if currency else "Price (USDT)"
    layout_dict = {
        "title": (
            f"{title_prefix} Price Simulations Comparison"
            if title_prefix
            else "Price Simulations Comparison"
        ),
        "title_x": 0.5,
        "title_xanchor": "center",
        "title_font_size": 28,
        "xaxis_title": "Timestamp",
        "yaxis_title": yaxis_title,
        "hovermode": "closest",
        "template": "plotly_white",
        "height": 800,
        "font": {"size": 12, "color": "black"},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "title_font_color": "black",
        "margin": {"l": 80, "r": 120, "t": 100, "b": 80},
    }

    # Add secondary y-axis if rebalance_magnitudes have data
    has_rebalance_data = (
        any(mag is not None for mag in rebalance_magnitudes.values())
        if rebalance_magnitudes
        else False
    )
    if has_rebalance_data:
        layout_dict["yaxis2"] = {
            "title": "Rebalance Size (USD)",
            "overlaying": "y",
            "side": "right",
            "rangemode": "tozero",
            "showgrid": False,
            "zeroline": False,
            "title_font_color": "black",
            "tickfont_color": "black",
        }
        layout_dict["legend"] = {
            "title": "Price",
            "x": 0.02,
            "y": 0.02,
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "black",
            "borderwidth": 1,
            "xanchor": "left",
            "yanchor": "bottom",
            "font": {"color": "black", "size": 12},
        }
        layout_dict["legend2"] = {
            "title": "Rebalances",
            "x": 0.98,
            "y": 0.02,
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "black",
            "borderwidth": 1,
            "xanchor": "right",
            "yanchor": "bottom",
            "font": {"color": "black", "size": 12},
        }
        layout_dict["margin"] = {"l": 80, "r": 120, "t": 100, "b": 80}
    else:
        layout_dict["legend"] = {
            "x": 0.02,
            "y": 0.02,
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "black",
            "borderwidth": 1,
            "xanchor": "left",
            "yanchor": "bottom",
            "font": {"color": "black", "size": 12},
        }

    price_fig.update_layout(**layout_dict)
    price_fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        title_font_color="black",
        tickfont_color="black",
    )
    # Set y-axis to start from 0 and go to 1.2x max price
    max_price = (
        market_price.max() if hasattr(market_price, "max") else max(market_price)
    )
    price_fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        title_font_color="black",
        tickfont_color="black",
        range=[0, 1.2 * max_price],
        selector={"overlaying": None},
    )

    # Create leverage plot
    leverage_fig = go.Figure()

    # Determine which columns to use based on leverage_timing
    lev_prefix = "lambdast" if leverage_timing == "After Rebalance" else "lambda"
    up_col = f"{lev_prefix}_up"
    down_col = f"{lev_prefix}_down"

    for idx, (name, df) in enumerate(results_dict.items()):
        color = TRACE_COLORS[idx % len(TRACE_COLORS)]

        # Plot UP leverage
        if up_col in df.columns:
            leverage_fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=df[up_col].values,
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    line={"color": color, "width": 2},
                    hovertemplate=(
                        f"<b>{name} UP</b><br>Time: %{{x|%H:%M:%S}}<br>Leverage: %{{y:.3f}}<extra></extra>"
                        if show_hover
                        else None
                    ),
                    hoverinfo="skip" if not show_hover else None,
                )
            )

        # Plot DOWN leverage (negated for display)
        if down_col in df.columns:
            leverage_fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=-df[down_col].values,
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    line={"color": color, "width": 2},
                    customdata=df[down_col].values,
                    hovertemplate=(
                        f"<b>{name} DOWN</b><br>Time: %{{x|%H:%M:%S}}<br>Leverage: %{{customdata:.3f}}<extra></extra>"
                        if show_hover
                        else None
                    ),
                    hoverinfo="skip" if not show_hover else None,
                )
            )

    leverage_fig.update_layout(
        title=(
            f"{title_prefix} Leverage {leverage_timing}"
            if title_prefix
            else f"Leverage {leverage_timing}"
        ),
        title_x=0.5,
        title_xanchor="center",
        title_font_size=28,
        xaxis_title="Timestamp",
        yaxis_title="Leverage (λ)",
        hovermode="closest",
        template="plotly_white",
        height=600,
        font={"size": 12, "color": "black"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={
            "x": 0.98,
            "y": 0.02,
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "black",
            "borderwidth": 1,
            "xanchor": "right",
            "yanchor": "bottom",
            "font": {"color": "black", "size": 12},
            "tracegroupgap": 0,
        },
        margin={"l": 80, "r": 120, "t": 100, "b": 80},
    )
    leverage_fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        title_font_color="black",
        tickfont_color="black",
    )
    leverage_fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        title_font_color="black",
        tickfont_color="black",
        zeroline=True,
        zerolinecolor="gray",
        zerolinewidth=1,
    )

    # Add lambda boundary lines
    for threshold, label in [(lambda_upper, "λ_upper"), (lambda_lower, "λ_lower")]:
        for sign in (1, -1):
            leverage_fig.add_hline(
                y=sign * threshold,
                line={"color": "gray", "width": 1, "dash": "dot"},
                annotation_text=f"{label} = {sign * threshold:g}",
                annotation_position="top left",
                annotation_font_color="gray",
                annotation_font_size=11,
            )

    return price_fig, leverage_fig


def plot_orderbook_depth(
    results_dict,
    x_axis=None,
    show_hover=True,
    title_prefix="",
    resample_freq=None,
):
    """Plot bid and ask depth over time from simulation results in two subplots.

    Args:
        results_dict: Dict of result dataframes (must contain 'depth_bid' and 'depth_ask' columns)
        x_axis: X-axis values (if None, uses list of indices). Should be timestamps.
        show_hover: Whether to show hover info
        title_prefix: Prefix for plot title (e.g., "BTC 30s")
        resample_freq: Resample frequency (e.g., '30s', '1min', '15min'). None = no resampling.

    Returns:
        Plotly figure object
    """
    first_df = next(iter(results_dict.values()))
    if "timestamp" in first_df.columns:
        result_timestamps = first_df["timestamp"].values
        if x_axis is None:
            x_axis = result_timestamps

    # Resample data if requested
    if resample_freq and resample_freq != "raw" and x_axis is not None and len(x_axis) > 0:
        resampled_results_dict = {}
        for name, df in results_dict.items():
            df_temp = df.copy()
            if "timestamp" in df_temp.columns:
                df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"], utc=True)
                df_temp.set_index("timestamp", inplace=True)
            else:
                df_temp["timestamp"] = pd.to_datetime(x_axis, utc=True)
                df_temp.set_index("timestamp", inplace=True)
            df_resampled = df_temp.resample(resample_freq).last()
            resampled_results_dict[name] = df_resampled.reset_index(drop=True)

        results_dict = resampled_results_dict
        temp_df = pd.DataFrame({"timestamp": x_axis})
        temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"], utc=True)
        temp_df.set_index("timestamp", inplace=True)
        resampled = temp_df.resample(resample_freq).last()
        x_axis = resampled.index.values

    first_df = next(iter(results_dict.values()))
    if x_axis is None:
        x_axis = list(range(len(first_df)))

    # Create subplots: 2 rows, 1 column
    depth_fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Bid Depth", "Ask Depth"),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    for idx, (name, df) in enumerate(results_dict.items()):
        # Skip "No Orderbook" result
        if name == "No Orderbook":
            continue

        color = TRACE_COLORS[idx % len(TRACE_COLORS)]

        # Check if depth columns exist
        if "depth_bid" not in df.columns or "depth_ask" not in df.columns:
            continue

        # Plot bid depth (row 1)
        depth_fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=df["depth_bid"].values,
                mode="lines",
                name=name,
                line={"color": color, "width": 2},
                legendgroup=name,
                hovertemplate=(
                    f"<b>{name} Bid</b><br>Time: %{{x|%H:%M:%S}}<br>Depth: $%{{y:,.0f}}<extra></extra>"
                    if show_hover
                    else None
                ),
                hoverinfo="skip" if not show_hover else None,
            ),
            row=1,
            col=1,
        )

        # Plot ask depth (row 2)
        depth_fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=df["depth_ask"].values,
                mode="lines",
                name=name,
                line={"color": color, "width": 2},
                legendgroup=name,
                showlegend=False,
                hovertemplate=(
                    f"<b>{name} Ask</b><br>Time: %{{x|%H:%M:%S}}<br>Depth: $%{{y:,.0f}}<extra></extra>"
                    if show_hover
                    else None
                ),
                hoverinfo="skip" if not show_hover else None,
            ),
            row=2,
            col=1,
        )

    depth_fig.update_layout(
        title=(
            f"{title_prefix} Orderbook Depth Over Time"
            if title_prefix
            else "Orderbook Depth Over Time"
        ),
        title_x=0.5,
        title_xanchor="center",
        title_font_size=28,
        hovermode="closest",
        template="plotly_white",
        height=800,
        font={"size": 12, "color": "black"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={
            "x": 0.02,
            "y": 0.98,
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "black",
            "borderwidth": 1,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"color": "black", "size": 12},
            "tracegroupgap": 0,
        },
        margin={"l": 80, "r": 80, "t": 100, "b": 80},
    )

    # Update x-axes
    depth_fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        title_font_color="black",
        tickfont_color="black",
        title_text="Timestamp",
        row=2,
        col=1,
    )

    # Update y-axes
    depth_fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        title_font_color="black",
        tickfont_color="black",
        title_text="Depth (USD)",
        rangemode="tozero",
    )

    return depth_fig
