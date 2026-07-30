"""
VLT Simulation Dashboard
Run:
streamlit run simulations/dashboard.py
"""

import json
import sys
import uuid
from pathlib import Path

# Add project root to path before importing local modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from functions.data_processing import FREQUENCIES
from functions.orderbook import Orderbook
from functions.plot_utils import plot_results, resample_data
from functions.price_spillover_simulations import run_simulation


def frequency_to_seconds(freq):
    """Convert frequency string to seconds for comparison."""
    if freq == "Tick":
        return 0
    if freq.endswith("ms"):
        return float(freq[:-2]) / 1000
    if freq.endswith("s"):
        return float(freq[:-1])
    if freq.endswith("min"):
        return float(freq[:-3]) * 60
    return float('inf')

# Persistence setup
CONFIG_DIR = Path(__file__).parent.parent / ".dashboard_config"
CONFIG_FILE = CONFIG_DIR / "dashboard_state.json"
DEFAULTS_FILE = CONFIG_DIR / "dashboard_defaults.json"

def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
    return None

def load_defaults():
    if DEFAULTS_FILE.exists():
        try:
            with open(DEFAULTS_FILE, 'r') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
    return None

def save_config(orderbooks, currency, frequency, lambda_target, lambda_value, lambda_upper, lambda_lower, show_hover, show_markers, leverage_timing, include_baseline):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "orderbooks": orderbooks,
        "currency": currency,
        "frequency": frequency,
        "lambda_target": lambda_target,
        "lambda_value": lambda_value,
        "lambda_upper": lambda_upper,
        "lambda_lower": lambda_lower,
        "show_hover": show_hover,
        "show_markers": show_markers,
        "leverage_timing": leverage_timing,
        "include_baseline": include_baseline,
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def save_as_defaults(orderbooks, currency, frequency, lambda_target, lambda_value, lambda_upper, lambda_lower, show_hover, show_markers, leverage_timing, include_baseline):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    defaults = {
        "orderbooks": orderbooks,
        "currency": currency,
        "frequency": frequency,
        "lambda_target": lambda_target,
        "lambda_value": lambda_value,
        "lambda_upper": lambda_upper,
        "lambda_lower": lambda_lower,
        "show_hover": show_hover,
        "show_markers": show_markers,
        "leverage_timing": leverage_timing,
        "include_baseline": include_baseline,
    }
    with open(DEFAULTS_FILE, 'w') as f:
        json.dump(defaults, f, indent=2)

st.set_page_config(layout="wide", page_title="Test Bench Dashboard")

# Load persisted config and defaults
saved_config = load_config()
saved_defaults = load_defaults()

# Initialize session state
if "custom_orderbooks" not in st.session_state:
    st.session_state.custom_orderbooks = {}
if "plot_resample_freq" not in st.session_state:
    st.session_state.plot_resample_freq = "15s"

def reset_plot_resample():
    """Reset plot resample frequency when settings change."""
    st.session_state.plot_resample_freq = "15s"

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    # Currency and Frequency selection
    st.subheader("Simulation Parameters")
    default_currency = saved_config.get("currency") if saved_config else (saved_defaults.get("currency") if saved_defaults else "btc")
    default_frequency = saved_config.get("frequency") if saved_config else (saved_defaults.get("frequency") if saved_defaults else "30s")

    available_currencies = ["btc", "sushi", "eth"]
    default_currency_index = available_currencies.index(default_currency) if default_currency in available_currencies else 0
    currency = st.radio(
        "Currency",
        available_currencies,
        horizontal=True,
        index=default_currency_index,
        on_change=reset_plot_resample
    )

    default_freq_index = FREQUENCIES.index(default_frequency) if default_frequency in FREQUENCIES else FREQUENCIES.index("30s")
    frequency = st.selectbox(
        "Simulation Frequency",
        FREQUENCIES,
        index=default_freq_index,
        on_change=reset_plot_resample
    )
    default_show_hover = (saved_config.get("show_hover") if saved_config else None) or (saved_defaults.get("show_hover") if saved_defaults else False)
    show_hover = st.checkbox("Show hover info", value=default_show_hover)

    # Plot resampling
    st.subheader("Plot Display")
    # Build resample options: raw or frequencies strictly greater than simulation frequency
    all_resample_options = ["raw"] + FREQUENCIES + ["5min", "15min"]
    sim_freq_seconds = frequency_to_seconds(frequency)
    resample_options = ["raw"] + [f for f in all_resample_options[1:] if frequency_to_seconds(f) > sim_freq_seconds]
    resample_options = list(dict.fromkeys(resample_options))  # Remove duplicates while preserving order

    # Use session state value if it's in the available options, otherwise default to "raw"
    if st.session_state.plot_resample_freq in resample_options:
        default_resample_index = resample_options.index(st.session_state.plot_resample_freq)
    else:
        default_resample_index = 0  # Default to "raw"

    plot_resample_freq = st.selectbox(
        "Resample for plot clarity",
        resample_options,
        index=default_resample_index,
        help="Use 'raw' to show all data points. Only shows resample frequencies coarser than simulation frequency."
    )
    st.session_state.plot_resample_freq = plot_resample_freq

    default_show_markers = (saved_config.get("show_markers") if saved_config else None) or (saved_defaults.get("show_markers") if saved_defaults else True)
    show_markers = st.checkbox("Show markers on lines", value=default_show_markers)

    default_leverage_timing = (saved_config.get("leverage_timing") if saved_config else None) or (saved_defaults.get("leverage_timing") if saved_defaults else "After Rebalance")
    leverage_timing = st.selectbox(
        "Leverage plot",
        ["Before Rebalance", "After Rebalance"],
        index=1 if default_leverage_timing == "After Rebalance" else 0,
        help="Show leverage (λ) before or after the rebalance trade is applied."
    )

    # Rebalancing parameters
    st.subheader("Rebalancing Strategy")
    default_lambda_target = (saved_config.get("lambda_target") if saved_config else None) or (saved_defaults.get("lambda_target") if saved_defaults else False)
    default_lambda_value = (saved_config.get("lambda_value") if saved_config else None) or (saved_defaults.get("lambda_value") if saved_defaults else 1.5)
    default_lambda_upper = (saved_config.get("lambda_upper") if saved_config else None) or (saved_defaults.get("lambda_upper") if saved_defaults else 4.0)
    default_lambda_lower = (saved_config.get("lambda_lower") if saved_config else None) or (saved_defaults.get("lambda_lower") if saved_defaults else 1.25)

    # Strategy toggle
    rebalancing_mode = st.radio(
        "Strategy",
        ["Nearest Boundary", "Target Leverage"],
        index=1 if default_lambda_target else 0,
        horizontal=True,
        label_visibility="collapsed",
        on_change=reset_plot_resample
    )
    lambda_target = rebalancing_mode == "Target"

    # Target value field (only show if Target mode)
    if lambda_target:
        lambda_value = st.number_input(
            "Lambda target value",
            value=default_lambda_value,
            min_value=0.0,
            step=0.1
        )
    else:
        lambda_value = None

    # Lambda boundaries
    st.write("Lambda boundaries")
    bound_col1, bound_col2 = st.columns(2)

    with bound_col1:
        lambda_lower = st.number_input(
            "Lower boundary",
            value=default_lambda_lower,
            min_value=0.1,
            max_value=10.0,
            step=0.01
        )
    with bound_col2:
        lambda_upper = st.number_input(
            "Upper boundary",
            value=default_lambda_upper,
            min_value=0.1,
            max_value=10.0,
            step=0.01
        )

    # Orderbook selection
    st.subheader("Orderbook Selection")
    default_include_baseline = (saved_config.get("include_baseline") if saved_config else None) or (saved_defaults.get("include_baseline") if saved_defaults else True)
    include_baseline = st.checkbox("Include baseline (no orderbook)", value=default_include_baseline, on_change=reset_plot_resample)

    # Initialize orderbooks list if not exists
    if "orderbooks_list" not in st.session_state:
        if saved_config and "orderbooks" in saved_config:
            st.session_state.orderbooks_list = saved_config["orderbooks"]
        elif saved_defaults and "orderbooks" in saved_defaults:
            st.session_state.orderbooks_list = saved_defaults["orderbooks"]
        else:
            st.session_state.orderbooks_list = [
                {"_id": str(uuid.uuid4()), "name": "Deep Narrow Tight", "depth_bid": 50_000_000, "depth_ask": 50_000_000, "width_bid": 1.0, "width_ask": 1.0, "spread_bid": 0.005, "spread_ask": 0.005, "k_bid": 1, "k_ask": 1},
                {"_id": str(uuid.uuid4()), "name": "Deep Narrow Broad", "depth_bid": 50_000_000, "depth_ask": 50_000_000, "width_bid": 1.0, "width_ask": 1.0, "spread_bid": 0.05, "spread_ask": 0.05, "k_bid": 1, "k_ask": 1},
                {"_id": str(uuid.uuid4()), "name": "Deep Wide Tight", "depth_bid": 50_000_000, "depth_ask": 50_000_000, "width_bid": 10.0, "width_ask": 10.0, "spread_bid": 0.005, "spread_ask": 0.005, "k_bid": 1, "k_ask": 1},
                {"_id": str(uuid.uuid4()), "name": "Shallow Narrow Tight", "depth_bid": 5_000_000, "depth_ask": 5_000_000, "width_bid": 1.0, "width_ask": 1.0, "spread_bid": 0.005, "spread_ask": 0.005, "k_bid": 1, "k_ask": 1},
            ]

    # Display orderbook rows
    for idx, orderbook in enumerate(st.session_state.orderbooks_list):
        ob_id = orderbook.get("_id", str(uuid.uuid4()))
        if "_id" not in orderbook:
            orderbook["_id"] = ob_id

        # Name row with reorder and delete buttons
        col0, col1, col2, col3, col4 = st.columns([0.4, 4.6, 0.8, 0.8, 0.8])
        with col0:
            st.markdown(f"**{idx + 1}**")
        with col1:
            orderbook["name"] = st.text_input(
                "Name",
                value=orderbook["name"],
                key=f"name_{ob_id}",
                label_visibility="collapsed",
                placeholder="Orderbook name"
            )
        with col2:
            if idx > 0 and st.button("↑", key=f"up_{ob_id}", help="Move up"):
                st.session_state.orderbooks_list[idx], st.session_state.orderbooks_list[idx - 1] = (
                    st.session_state.orderbooks_list[idx - 1],
                    st.session_state.orderbooks_list[idx]
                )
                st.rerun()
        with col3:
            if idx < len(st.session_state.orderbooks_list) - 1 and st.button("↓", key=f"down_{ob_id}", help="Move down"):
                st.session_state.orderbooks_list[idx], st.session_state.orderbooks_list[idx + 1] = (
                    st.session_state.orderbooks_list[idx + 1],
                    st.session_state.orderbooks_list[idx]
                )
                st.rerun()
        with col4:
            if st.button("🗑️", key=f"delete_{ob_id}", help="Delete this orderbook"):
                st.session_state.orderbooks_list = [ob for ob in st.session_state.orderbooks_list if ob.get("_id") != ob_id]
                st.rerun()

        # Parameters row (Depth bid/ask, Width bid/ask, Spread bid/ask, k bid/ask)
        param_col1, param_col2, param_col3, param_col4 = st.columns([1.4, 1.0, 1.0, 0.9])

        with param_col1:
            st.markdown("**Depth (M USD)**")
            depth_bid_val = orderbook.get("depth_bid", 50_000_000) / 1_000_000
            depth_ask_val = orderbook.get("depth_ask", 50_000_000) / 1_000_000
            depth_input = st.text_input(
                "Depth",
                value=f"{depth_bid_val:.1f}, {depth_ask_val:.1f}",
                key=f"depth_{ob_id}",
                label_visibility="collapsed",
                placeholder="e.g., 50, 50 (bid, ask)"
            )

            try:
                parts = [float(p.strip()) * 1_000_000 for p in depth_input.split(",")]
                if len(parts) == 2:
                    orderbook["depth_bid"] = int(parts[0])
                    orderbook["depth_ask"] = int(parts[1])
                elif len(parts) == 1:
                    orderbook["depth_bid"] = int(parts[0])
                    orderbook["depth_ask"] = int(parts[0])
            except ValueError:
                pass

        with param_col2:
            st.markdown("**Width (%)**")
            width_bid_val = orderbook.get("width_bid", 1.0)
            width_ask_val = orderbook.get("width_ask", 1.0)
            width_input = st.text_input(
                "Width",
                value=f"{width_bid_val}, {width_ask_val}",
                key=f"width_{ob_id}",
                label_visibility="collapsed",
                placeholder="e.g., 1, 1 (bid, ask)"
            )
            st.caption("bid, ask (e.g., 1 = 1%)")

            try:
                parts = [float(p.strip()) for p in width_input.split(",")]
                if len(parts) == 2:
                    orderbook["width_bid"] = parts[0]
                    orderbook["width_ask"] = parts[1]
                elif len(parts) == 1:
                    orderbook["width_bid"] = parts[0]
                    orderbook["width_ask"] = parts[0]
            except ValueError:
                pass

        with param_col3:
            st.markdown("**Spread (%)**")
            spread_bid_val = orderbook.get("spread_bid", 0.005)
            spread_ask_val = orderbook.get("spread_ask", 0.005)
            spread_input = st.text_input(
                "Spread",
                value=f"{spread_bid_val}, {spread_ask_val}",
                key=f"spread_{ob_id}",
                label_visibility="collapsed",
                placeholder="e.g., 0.005, 0.005 (bid, ask)"
            )
            st.caption("bid, ask (e.g., 0.005 = 0.5 bps)")

            try:
                parts = [float(p.strip()) for p in spread_input.split(",")]
                if len(parts) == 2:
                    orderbook["spread_bid"] = parts[0]
                    orderbook["spread_ask"] = parts[1]
                elif len(parts) == 1:
                    orderbook["spread_bid"] = parts[0]
                    orderbook["spread_ask"] = parts[0]
            except ValueError:
                pass

        with param_col4:
            st.markdown("**k (Curvature)**")
            k_bid_val = orderbook.get("k_bid", 1)
            k_ask_val = orderbook.get("k_ask", 1)
            k_input = st.text_input(
                "k",
                value=f"{k_bid_val}, {k_ask_val}",
                key=f"k_{ob_id}",
                label_visibility="collapsed",
                placeholder="e.g., 1, 1 (bid, ask)"
            )
            st.caption("bid, ask (k=1: linear)")

            try:
                parts = [float(p.strip()) for p in k_input.split(",")]
                if len(parts) == 2:
                    orderbook["k_bid"] = parts[0]
                    orderbook["k_ask"] = parts[1]
                elif len(parts) == 1:
                    orderbook["k_bid"] = parts[0]
                    orderbook["k_ask"] = parts[0]
            except ValueError:
                pass

        st.divider()

    # Add row button
    if len(st.session_state.orderbooks_list) < 8 and st.button("+ Add Orderbook", type="primary"):
        st.session_state.orderbooks_list.append({
            "_id": str(uuid.uuid4()),
            "name": f"Orderbook {len(st.session_state.orderbooks_list) + 1}",
            "depth_bid": 50_000_000,
            "depth_ask": 50_000_000,
            "width_bid": 1.0,
            "width_ask": 1.0,
            "spread_bid": 0.005,
            "spread_ask": 0.005,
            "k_bid": 1,
            "k_ask": 1,
        })
        st.rerun()

    # Settings management buttons
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save as Default", width='stretch'):
            save_as_defaults(st.session_state.orderbooks_list, currency, frequency, lambda_target, lambda_value, lambda_upper, lambda_lower, show_hover, show_markers, leverage_timing, include_baseline)
            st.success("Settings saved as default!")

    with col2:
        if st.button("🔄 Reset to Default", width='stretch'):
            if saved_defaults:
                st.session_state.orderbooks_list = saved_defaults.get("orderbooks", st.session_state.orderbooks_list)
                st.rerun()
            else:
                st.info("No defaults saved yet")

    # Build selected_orderbooks from the list
    selected_orderbooks = {}
    if include_baseline:
        selected_orderbooks["Baseline (No Orderbook)"] = None

    for ob_dict in st.session_state.orderbooks_list:
        if ob_dict["name"]:  # Only include if name is not empty
            ob = Orderbook(
                name=ob_dict["name"],
                depth_bid=ob_dict.get("depth_bid", 50_000_000),
                depth_ask=ob_dict.get("depth_ask", 50_000_000),
                width_bid=ob_dict.get("width_bid", 1.0),
                width_ask=ob_dict.get("width_ask", 1.0),
                spread_bid=ob_dict.get("spread_bid", 0.005),
                spread_ask=ob_dict.get("spread_ask", 0.005),
                k_bid=ob_dict.get("k_bid", 1),
                k_ask=ob_dict.get("k_ask", 1),
            )
            selected_orderbooks[ob.name] = ob

    # Save configuration
    save_config(st.session_state.orderbooks_list, currency, frequency, lambda_target, lambda_value, lambda_upper, lambda_lower, show_hover, show_markers, leverage_timing, include_baseline)

# Main content
if not selected_orderbooks:
    st.warning("Please select at least one orderbook to run the simulation.")
else:
    # Create placeholders for plots at the top
    plot_placeholder = st.empty()
    leverage_placeholder = st.empty()

    # Run simulation button
    if st.button("Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                try:
                    # Load pre-processed data
                    if frequency == "Tick":
                        filepath = f"dissertation_data/token_dataframes/{currency}_tick_processed.parquet"
                    else:
                        filepath = f"dissertation_data/token_dataframes/{currency}_{frequency}_processed.parquet"

                    binance_data = pd.read_parquet(filepath)
                    binance_data["timestamp"] = pd.to_datetime(binance_data["timestamp"], utc=True)

                    results = {}

                    # Run simulations
                    st.info("Orderbook Parameters:")
                    for orderbook_name, orderbook in selected_orderbooks.items():
                        if orderbook is not None:
                            st.write(f"**{orderbook_name}** → Depth: ({orderbook.depth_bid:,}, {orderbook.depth_ask:,}), Width: ({orderbook.width_bid}, {orderbook.width_ask}), Spread: ({orderbook.spread_bid}, {orderbook.spread_ask})")

                        result = run_simulation(
                            lambda_target=lambda_value,
                            lambda_upper=lambda_upper,
                            lambda_lower=lambda_lower,
                            orderbook=orderbook,
                            prepared_data=binance_data,
                            timestamps=binance_data["timestamp"].values
                        )
                        results[orderbook_name] = result

                    # Resample data for plotting if requested
                    plot_data = binance_data.copy()
                    resampled_results = {}
                    rebalance_magnitudes = {}

                    if plot_resample_freq != "raw":
                        # Resample binance_data (market price and timestamp)
                        plot_data = resample_data(plot_data, plot_resample_freq)

                        # For each result, add timestamp and resample, keeping only price_multiplier
                        for name, result in results.items():
                            result_with_ts = result.copy()
                            result_with_ts['timestamp'] = binance_data['timestamp'].values
                            resampled = resample_data(result_with_ts, plot_resample_freq)
                            # Keep only the columns we need
                            resampled_results[name] = resampled[
                                ['timestamp', 'price_multiplier', 'lambdast_up', 'lambdast_down', 'lambda_up', 'lambda_down']
                            ].reset_index(drop=True)

                            # Combined rebalance size = |UP delta| + |DOWN delta|. actual_total_delta is
                            # the *signed* sum of both sides, which largely cancels since UP and DOWN
                            # rebalance in opposite directions on the same move — using it produced
                            # near-zero, sign-flipping spikes. Summing the absolute values per side gives
                            # the actual combined trade size, always >= 0.
                            result_with_ts['rebalance_magnitude'] = (
                                result_with_ts['actual_delta_up'].abs() + result_with_ts['actual_delta_down'].abs()
                            )
                            magnitude_by_bucket = (
                                result_with_ts[['timestamp', 'rebalance_magnitude']]
                                .set_index('timestamp')
                                .resample(plot_resample_freq)
                                .sum()
                                .reset_index()
                            )
                            rebalance_magnitudes[name] = magnitude_by_bucket['rebalance_magnitude'].values
                    else:
                        # For raw data, just extract price_multiplier from each result
                        for name, result in results.items():
                            resampled_results[name] = result[
                                ['price_multiplier', 'lambdast_up', 'lambdast_down', 'lambda_up', 'lambda_down']
                            ].reset_index(drop=True)
                            rebalance_magnitudes[name] = (
                                result['actual_delta_up'].abs() + result['actual_delta_down'].abs()
                            ).values

                    # Plot simulations using shared plotting utility
                    title_freq = frequency.replace('min', 'min ')
                    title_prefix = f'{currency.upper()} {title_freq}'

                    # Prepare data for plotting: convert price_multiplier to simulated_price
                    plot_results_dict = {}
                    for name, result in resampled_results.items():
                        plot_result = result.copy()
                        plot_result['simulated_price'] = plot_data['price'].values * result['price_multiplier'].values
                        plot_results_dict[name] = plot_result

                    price_fig, leverage_fig = plot_results(
                        plot_results_dict,
                        market_price=plot_data['price'].values,
                        x_axis=plot_data['timestamp'].values,
                        lambda_upper=lambda_upper,
                        lambda_lower=lambda_lower,
                        show_hover=show_hover,
                        show_markers=show_markers,
                        leverage_timing=leverage_timing,
                        title_prefix=title_prefix,
                        rebalance_magnitudes=rebalance_magnitudes,
                        currency=currency.upper()
                    )

                    # Display results in placeholder at top
                    plot_placeholder.plotly_chart(price_fig, width='stretch')
                    leverage_placeholder.plotly_chart(leverage_fig, width='stretch')

                    st.success("Simulation completed!")

                except FileNotFoundError:
                    st.error(
                        f"Pre-processed data not found for {currency.upper()} at {frequency} frequency.\n\n"
                        f"Available frequencies: Tick, 15s, 30s, 1min\n\n"
                        f"Please run `python new_code/data_processing.py` to prepare the data."
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error running simulation: {e!s}")
