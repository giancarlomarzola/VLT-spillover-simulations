"""Default plot functions.

Simulation results (plot_prices, plot_leverage, plot_orderbook_depth): interactive Plotly
figures of the output of core.simulation.run_simulation, with time on the x-axis. Results
are DataFrames indexed by time: a DatetimeIndex (historical data) is plotted as dates, a
TimedeltaIndex (e.g. a GBM path) as time elapsed since the start of the simulation. Each
function takes either a single results DataFrame or a dict of {label: DataFrame} to compare
several runs (e.g. different orderbooks) on the same price path.

Orderbooks (plot_orderbooks): static matplotlib figure of what one or more orderbooks look
like (cumulative depth against price deviation).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.ticker import FuncFormatter
from plotly.subplots import make_subplots

from core.orderbook import Orderbook

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


def _rebalance_trace(delta, resample_freq, unit_from, name, color, show_hover, x_hover):
    """
    Stem trace of rebalance sizes (absolute trade size per step, summed per period when
    resampling) on the secondary axis, hidden until clicked in the legend.

    delta: Series of signed rebalance trades. unit_from: index of the main trace, so both
    share the same time unit. Returns None if there are no rebalances.
    """
    size = delta.abs()
    if resample_freq and resample_freq != "raw":
        size = size.resample(resample_freq).sum()
    size = size[size > 0]
    if not len(size):
        return None

    rx, _, _ = _time_axis(size.index, unit_from=unit_from)
    stem_x, stem_y = [], []
    for xi, yi in zip(rx, size.to_numpy()):
        stem_x += [xi, xi, None]
        stem_y += [0, yi, None]
    return go.Scatter(
        x=stem_x,
        y=stem_y,
        mode="lines",
        name=f"{name} rebalances",
        line={"color": color, "width": 2},
        yaxis="y2",
        legend="legend2",
        visible="legendonly",
        **_hover(show_hover, f"<b>{name} rebalance</b><br>Time: {x_hover}<br>Size: $%{{y:,.0f}}<extra></extra>"),
    )


def _rebalance_layout(fig, side):
    """Add the secondary axis and the legend (bottom 'left' or 'right') for rebalance traces."""
    fig.update_layout(
        # Always show the main legend, even with a single main trace: when Plotly hides it,
        # clicks on the rebalance legend do nothing in plotly.js 2.x (bundled by VS Code)
        showlegend=True,
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
            "x": 0.98 if side == "right" else 0.02,
            "y": 0.02,
            "xanchor": side,
            "yanchor": "bottom",
        },
    )


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

        trace = _rebalance_trace(
            df["actual_total_delta"], resample_freq, resampled.index, name, color, show_hover, x_hover
        )
        if trace is not None:
            has_rebalances = True
            fig.add_trace(trace)

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
        _rebalance_layout(fig, side="right")
    return fig


def plot_leverage(results, vletps, after_rebalance=False, resample_freq=None, show_hover=True, title_prefix=""):
    """
    Plot the leverage of each product over time, with its leverage band.

    Short products are drawn below zero, long products above.

    The size of each product's own rebalance trades is available on a secondary axis;
    those traces are hidden until clicked in the legend.

    Args:
        results: Results DataFrame, or dict of {label: DataFrame}
        vletps: The VLETP (or list of VLETPs) that were simulated
        after_rebalance: Plot leverage after each step's rebalance instead of before it
        resample_freq: Resample frequency (e.g. '30s', '1min'). None = no resampling.
            Leverage takes the last value in each period, rebalance sizes are summed.
        show_hover: Whether to show hover info
        title_prefix: Prefix for the plot title (e.g. "GBM")
    """
    results = _as_dict(results)
    vletps = vletps if isinstance(vletps, (list, tuple)) else [vletps]
    prefix = "leverage_after" if after_rebalance else "leverage"
    fig = go.Figure()
    x_title = "Time"
    has_rebalances = False

    trace_idx = 0
    for name, df in results.items():
        resampled = _resample(df, resample_freq)
        x, x_title, x_hover = _time_axis(resampled.index)
        for product in vletps:
            label = product.name if len(results) == 1 else f"{name} - {product.name}"
            color = TRACE_COLORS[trace_idx % len(TRACE_COLORS)]
            leverage = resampled[f"{prefix}_{product.name}"]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=product.direction * leverage,
                    mode="lines",
                    name=label,
                    line={"color": color, "width": 2},
                    **_hover(show_hover, f"<b>{label}</b><br>Time: {x_hover}<br>Leverage: %{{y:.3f}}<extra></extra>"),
                )
            )
            trace = _rebalance_trace(
                df[f"actual_delta_{product.name}"], resample_freq, resampled.index, label, color, show_hover, x_hover
            )
            if trace is not None:
                has_rebalances = True
                fig.add_trace(trace)
            trace_idx += 1

    timing = "After Rebalance" if after_rebalance else "Before Rebalance"
    _layout(fig, f"Leverage {timing}", title_prefix, height=600)
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Leverage (λ)",
        legend={**_LEGEND, "title": "Leverage", "x": 0.98, "y": 0.02, "xanchor": "right", "yanchor": "bottom"},
    )
    fig.update_yaxes(zeroline=True, zerolinecolor="gray", zerolinewidth=1)
    if has_rebalances:
        _rebalance_layout(fig, side="left")

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


def _orderbook_depth_curve(x, orderbook):
    """
    Cumulative depth f(z) = D * t/(k + (1-k)*t) of an orderbook at price deviations x (%).

    Negative x is the bid side, positive x the ask side. Uses the full (undepleted) depth.
    """
    depth_bid, depth_ask = orderbook.depth_bid_full, orderbook.depth_ask_full
    width_bid, width_ask = orderbook.width_bid * 100, orderbook.width_ask * 100
    spread_bid, spread_ask = orderbook.spread_bid * 100, orderbook.spread_ask * 100
    k_bid, k_ask = orderbook.k_bid, orderbook.k_ask

    t_bid = (-x - spread_bid) / (width_bid - spread_bid)
    bid_curve = depth_bid * t_bid / (k_bid + (1 - k_bid) * t_bid)

    t_ask = (x - spread_ask) / (width_ask - spread_ask)
    ask_curve = depth_ask * t_ask / (k_ask + (1 - k_ask) * t_ask)

    return np.where(
        x <= -width_bid,
        depth_bid,
        np.where(
            x <= -spread_bid,
            bid_curve,
            np.where(x < spread_ask, 0, np.where(x < width_ask, ask_curve, depth_ask)),
        ),
    )


def _format_side(bid, ask):
    """Format a per-side parameter: one number if both sides match, else bid/ask."""
    return f"{bid:g}" if bid == ask else f"{bid:g}/{ask:g}"


def _grid_shape(n, rows, cols):
    """Rows and columns for n panels. Missing values are derived: by default up to 2 columns."""
    if rows is None and cols is None:
        cols = min(n, 2)
    if cols is None:
        cols = -(-n // rows)
    if rows is None:
        rows = -(-n // cols)
    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be at least 1, got rows={rows}, cols={cols}")
    if rows * cols < n:
        raise ValueError(f"A {rows}x{cols} grid has room for {rows * cols} orderbooks, got {n}")
    return rows, cols


def plot_orderbooks(
    orderbooks,
    rows=None,
    cols=None,
    show_parameters=True,
    title=None,
    figsize=None,
    x_limit=None,
    y_limit=None,
):
    """
    Plot what one or more orderbooks look like: cumulative bid and ask depth against the
    price deviation from mid, one panel per orderbook.

    All panels share the same axis ranges, so the orderbooks can be compared by eye. The
    full depth is plotted, regardless of any depletion from earlier simulations.

    Args:
        orderbooks: An Orderbook, or a list of Orderbooks
        rows: Number of rows in the grid of panels. None = derived from cols.
        cols: Number of columns in the grid of panels. None = derived from rows, or
            up to 2 columns if rows is not given either.
        show_parameters: Whether to print each orderbook's parameters (depth, width,
            spread and, if not 1, k) under its name. False = name only.
        title: Title above all panels (optional)
        figsize: Figure size (width, height) in inches. None = scaled to the grid.
        x_limit: Range of the x axis, in % price deviation. A number = from -x_limit to
            x_limit, a (min, max) tuple = that range. None = 1.2x the widest orderbook.
        y_limit: Range of the y axis, in USD. A number = from 0 to y_limit, a (min, max)
            tuple = that range. None = 1.2x the deepest orderbook.

    Returns:
        The matplotlib Figure (show it with plt.show(), save it with fig.savefig(path))
    """
    orderbooks = [orderbooks] if isinstance(orderbooks, Orderbook) else list(orderbooks)
    if not orderbooks:
        raise ValueError("No orderbooks to plot")
    rows, cols = _grid_shape(len(orderbooks), rows, cols)

    if figsize is None:
        figsize = (max(5.25 * cols, 6.5), 3.2 * rows + 0.8)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, layout="constrained")
    axes = axes.flatten()

    if x_limit is None:
        x_limit = 1.2 * 100 * max(max(ob.width_bid, ob.width_ask) for ob in orderbooks)
    if y_limit is None:
        y_limit = 1.2 * max(max(ob.depth_bid_full, ob.depth_ask_full) for ob in orderbooks)
    x_min, x_max = x_limit if isinstance(x_limit, (tuple, list)) else (-x_limit, x_limit)
    y_min, y_max = y_limit if isinstance(y_limit, (tuple, list)) else (0, y_limit)
    x = np.linspace(x_min, x_max, 1000)

    for ax, ob in zip(axes, orderbooks):
        y = _orderbook_depth_curve(x, ob)

        bid_col = (153/255, 0, 0)
        ask_col = (20/255, 45/255, 80/255)

        ax.fill_between(x, y, where=(x <= 0), color=bid_col, alpha=0.8, 
                        label="Cumulative Bid", interpolate=True)
        ax.fill_between(x, y, where=(x >= 0), color=ask_col, alpha=0.8, 
                        label="Cumulative Ask", interpolate=True)
        ax.plot(x[x <= 0], y[x <= 0], color=bid_col, lw=0.7)
        ax.plot(x[x >= 0], y[x >= 0], color=ask_col, lw=0.7)

        parameters = (
            f"Depth=${_format_side(ob.depth_bid_full / 1e6, ob.depth_ask_full / 1e6)}M, "
            f"Width={_format_side(ob.width_bid * 100, ob.width_ask * 100)}%, "
            f"Spread={_format_side(ob.spread_bid * 100, ob.spread_ask * 100)}%"
        )
        if ob.k_bid != 1 or ob.k_ask != 1:
            parameters += f", k={_format_side(ob.k_bid, ob.k_ask)}"
        ax.set_title(f"{ob.name}\n{parameters}" if show_parameters else ob.name, fontsize=10)
        ax.set_xlabel("Price deviation (%)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel("Depth (USD millions)")
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:,.10g}"))
        #ax.grid(True, alpha=0.3)

    for ax in axes[len(orderbooks) :]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=10)
    return fig
