"""Plots for results of core.simulation.run_simulation, with time on the x-axis.

Results are DataFrames indexed by time: a DatetimeIndex (historical data) is plotted as
dates, a TimedeltaIndex (e.g. a GBM path) as time elapsed since the start of the simulation.

Every plot function takes either a single results DataFrame or a dict of
{label: DataFrame} to compare several runs (e.g. different orderbooks) on the same price path.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.plotting import TRACE_COLORS

_LEGEND = {
    "bgcolor": "rgba(255, 255, 255, 0.9)",
    "bordercolor": "black",
    "borderwidth": 1,
    "font": {"color": "black", "size": 12},
}
_AXIS = {
    "showgrid": True,
    "gridwidth": 1,
    "gridcolor": "lightgray",
    "title_font_color": "black",
    "tickfont_color": "black",
}
# (unit name, seconds per unit), largest first
_ELAPSED_UNITS = [("days", 86_400), ("hours", 3_600), ("minutes", 60), ("seconds", 1)]


def _as_dict(results):
    return {"Simulated": results} if isinstance(results, pd.DataFrame) else results


def _time_axis(index, unit_from=None):
    """
    Turn a results index into (x values, x-axis title, hover format for x).

    Elapsed time is plotted as a plain number in the largest unit that keeps the run at
    least 5 units long (e.g. a 3 hour run is shown in minutes), since Plotly cannot draw
    timedeltas directly. Pass unit_from (another index) to pick the unit from that index's
    length instead, so that two traces share the same unit.
    """
    if not isinstance(index, pd.TimedeltaIndex):
        return index, "Time", "%{x|%Y-%m-%d %H:%M:%S}"

    reference = (index if unit_from is None else unit_from).total_seconds()
    span = reference[-1] - reference[0] if len(reference) else 0.0
    unit, size = next(((u, s) for u, s in _ELAPSED_UNITS if span >= 5 * s), _ELAPSED_UNITS[-1])
    return index.total_seconds().to_numpy() / size, f"Time elapsed ({unit})", f"%{{x:,.2f}} {unit}"


def _resample(df, resample_freq):
    """Keep the last value in each period (e.g. '30s'). None or 'raw' leaves the data as is."""
    if not resample_freq or resample_freq == "raw":
        return df
    return df.resample(resample_freq).last().dropna(how="all")


def _hover(show_hover, template):
    return {"hovertemplate": template} if show_hover else {"hoverinfo": "skip"}


def _layout(fig, title, title_prefix, height):
    fig.update_layout(
        title=f"{title_prefix} {title}" if title_prefix else title,
        title_x=0.5,
        title_xanchor="center",
        title_font_size=28,
        title_font_color="black",
        hovermode="closest",
        template="plotly_white",
        height=height,
        font={"size": 12, "color": "black"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 80, "r": 120, "t": 100, "b": 80},
    )
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)


def plot_prices(results, resample_freq=None, show_hover=True, title_prefix=""):
    """
    Plot the raw price path (no VLETP impact) against the simulated price of each run.

    The size of each run's net rebalance trades is available on a secondary axis; those
    traces are hidden until clicked in the legend.

    Args:
        results: Results DataFrame, or dict of {label: DataFrame}
        resample_freq: Resample frequency (e.g. '30s', '1min'). None = no resampling.
            Prices take the last value in each period, rebalance sizes are summed.
        show_hover: Whether to show hover info
        title_prefix: Prefix for the plot title (e.g. "GBM")
    """
    results = _as_dict(results)
    fig = go.Figure()
    has_rebalances = False

    for idx, (name, df) in enumerate(results.items()):
        color = TRACE_COLORS[idx % len(TRACE_COLORS)]
        resampled = _resample(df, resample_freq)
        x, x_title, x_hover = _time_axis(resampled.index)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=resampled["simulated_price"],
                mode="lines",
                name=name,
                line={"color": color, "width": 2},
                **_hover(show_hover, f"<b>{name}</b><br>Time: {x_hover}<br>Price: $%{{y:,.2f}}<extra></extra>"),
            )
        )

        rebalance_size = df["actual_total_delta"].abs()
        if resample_freq and resample_freq != "raw":
            rebalance_size = rebalance_size.resample(resample_freq).sum()
        rebalance_size = rebalance_size[rebalance_size > 0]
        if len(rebalance_size):
            has_rebalances = True
            rx, _, _ = _time_axis(rebalance_size.index, unit_from=resampled.index)
            stem_x, stem_y = [], []
            for xi, size in zip(rx, rebalance_size.to_numpy()):
                stem_x += [xi, xi, None]
                stem_y += [0, size, None]
            fig.add_trace(
                go.Scatter(
                    x=stem_x,
                    y=stem_y,
                    mode="lines",
                    name=f"{name} rebalances",
                    line={"color": color, "width": 2},
                    yaxis="y2",
                    legend="legend2",
                    visible="legendonly",
                    **_hover(
                        show_hover,
                        f"<b>{name} rebalance</b><br>Time: {x_hover}<br>Size: $%{{y:,.0f}}<extra></extra>",
                    ),
                )
            )

    # Raw price last, so it is drawn on top
    raw = _resample(next(iter(results.values())), resample_freq)
    x, x_title, x_hover = _time_axis(raw.index)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=raw["raw_price"],
            mode="lines",
            name="Raw price (no VLETP impact)",
            line={"color": "black", "width": 2},
            **_hover(show_hover, f"<b>Raw price</b><br>Time: {x_hover}<br>Price: $%{{y:,.2f}}<extra></extra>"),
        )
    )

    _layout(fig, "Raw vs Simulated Price", title_prefix, height=800)
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Price",
        legend={**_LEGEND, "title": "Price", "x": 0.02, "y": 0.02, "xanchor": "left", "yanchor": "bottom"},
    )
    if has_rebalances:
        fig.update_layout(
            yaxis2={
                **_AXIS,
                "title": "Rebalance Size",
                "overlaying": "y",
                "side": "right",
                "rangemode": "tozero",
                "showgrid": False,
                "zeroline": False,
            },
            legend2={
                **_LEGEND,
                "title": "Rebalances",
                "x": 0.98,
                "y": 0.02,
                "xanchor": "right",
                "yanchor": "bottom",
            },
        )
    return fig


def plot_leverage(results, vletps, after_rebalance=False, resample_freq=None, show_hover=True, title_prefix=""):
    """
    Plot the leverage of each product over time, with its leverage band.

    Short products are drawn below zero, long products above.

    Args:
        results: Results DataFrame, or dict of {label: DataFrame}
        vletps: The VLETP (or list of VLETPs) that were simulated
        after_rebalance: Plot leverage after each step's rebalance instead of before it
        resample_freq: Resample frequency (e.g. '30s', '1min'). None = no resampling.
        show_hover: Whether to show hover info
        title_prefix: Prefix for the plot title (e.g. "GBM")
    """
    results = _as_dict(results)
    vletps = vletps if isinstance(vletps, (list, tuple)) else [vletps]
    prefix = "leverage_after" if after_rebalance else "leverage"
    fig = go.Figure()
    x_title = "Time"

    trace_idx = 0
    for name, df in results.items():
        resampled = _resample(df, resample_freq)
        x, x_title, x_hover = _time_axis(resampled.index)
        for product in vletps:
            label = product.name if len(results) == 1 else f"{name} - {product.name}"
            leverage = resampled[f"{prefix}_{product.name}"]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=product.direction * leverage,
                    mode="lines",
                    name=label,
                    line={"color": TRACE_COLORS[trace_idx % len(TRACE_COLORS)], "width": 2},
                    **_hover(show_hover, f"<b>{label}</b><br>Time: {x_hover}<br>Leverage: %{{y:.3f}}<extra></extra>"),
                )
            )
            trace_idx += 1

    timing = "After Rebalance" if after_rebalance else "Before Rebalance"
    _layout(fig, f"Leverage {timing}", title_prefix, height=600)
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Leverage (λ)",
        legend={**_LEGEND, "x": 0.98, "y": 0.02, "xanchor": "right", "yanchor": "bottom"},
    )
    fig.update_yaxes(zeroline=True, zerolinecolor="gray", zerolinewidth=1)

    # Leverage band boundaries, once per distinct level
    bounds = set()
    for product in vletps:
        bounds.add((product.direction * product.leverage_upper_bound, "λ_upper"))
        bounds.add((product.direction * product.leverage_lower_bound, "λ_lower"))
    for level, label in sorted(bounds):
        fig.add_hline(
            y=level,
            line={"color": "gray", "width": 1, "dash": "dot"},
            annotation_text=f"{label} = {level:g}",
            annotation_position="top left",
            annotation_font_color="gray",
            annotation_font_size=11,
        )
    return fig


def plot_orderbook_depth(results, resample_freq=None, show_hover=True, title_prefix=""):
    """
    Plot bid and ask depth over time in two subplots. Runs without an orderbook are skipped.

    Args:
        results: Results DataFrame, or dict of {label: DataFrame}
        resample_freq: Resample frequency (e.g. '30s', '1min'). None = no resampling.
        show_hover: Whether to show hover info
        title_prefix: Prefix for the plot title (e.g. "GBM")
    """
    results = _as_dict(results)
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Bid Depth", "Ask Depth"),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )
    x_title = "Time"

    for idx, (name, df) in enumerate(results.items()):
        if df["depth_bid"].isna().all():
            continue  # run had no orderbook
        color = TRACE_COLORS[idx % len(TRACE_COLORS)]
        resampled = _resample(df, resample_freq)
        x, x_title, x_hover = _time_axis(resampled.index)
        for row, (side, column) in enumerate((("Bid", "depth_bid"), ("Ask", "depth_ask")), start=1):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=resampled[column],
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=row == 1,
                    line={"color": color, "width": 2},
                    **_hover(
                        show_hover,
                        f"<b>{name} {side}</b><br>Time: {x_hover}<br>Depth: $%{{y:,.0f}}<extra></extra>",
                    ),
                ),
                row=row,
                col=1,
            )

    _layout(fig, "Orderbook Depth Over Time", title_prefix, height=800)
    fig.update_layout(
        legend={**_LEGEND, "x": 0.02, "y": 0.98, "xanchor": "left", "yanchor": "top", "tracegroupgap": 0},
    )
    fig.update_xaxes(title_text=x_title, row=2, col=1)
    fig.update_yaxes(title_text="Depth", rangemode="tozero")
    return fig
