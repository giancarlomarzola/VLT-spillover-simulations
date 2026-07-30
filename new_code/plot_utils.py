"""Shared plotting utilities for simulation results."""

import numpy as np
import plotly.graph_objects as go

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
    import pandas as pd

    # Resample data if requested
    if resample_freq and resample_freq != "raw" and x_axis is not None and len(x_axis) > 0:
        temp_df = pd.DataFrame(
            {"timestamp": x_axis, "market_price": market_price}
        )
        temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"], utc=True)
        temp_df.set_index("timestamp", inplace=True)
        resampled = temp_df.resample(resample_freq).last()
        x_axis = resampled.index.values
        market_price = resampled["market_price"].values

        # Resample each result dataframe
        resampled_results_dict = {}
        for name, df in results_dict.items():
            df_temp = df.copy()
            # Get original x_axis indices for this dataframe
            orig_len = len(df_temp)
            df_temp["timestamp"] = x_axis[: len(x_axis)]  # Will be adjusted
            # Re-align: we need to create a temp dataframe with the original timestamps
            original_x = (
                pd.Series(x_axis).iloc[:orig_len].values
                if len(x_axis) >= orig_len
                else x_axis
            )
            df_temp["timestamp"] = original_x
            df_temp.set_index("timestamp", inplace=True)
            df_resampled = df_temp.resample(resample_freq).last()
            resampled_results_dict[name] = df_resampled.reset_index(drop=True)

        results_dict = resampled_results_dict
    # Get data dimensions
    first_df = next(iter(results_dict.values()))
    if x_axis is None:
        x_axis = list(range(len(first_df)))

    # Determine which price to use as baseline
    if market_price is None:
        if "raw_price" in first_df.columns:
            market_price = first_df["raw_price"].values
        else:
            raise ValueError(
                "market_price must be provided or 'raw_price' must be in dataframes"
            )

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
                    name=f"{name} UP",
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
                    name=f"{name} DOWN",
                    line={"color": color, "width": 2, "dash": "6 3"},
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
