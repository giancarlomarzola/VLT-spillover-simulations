import sys
import json
import uuid
from pathlib import Path

"""
TO RUN THE DASHBOARD USE THIS COMMAND IN TERMINAL: 
streamlit run new_code/dashboard.py
"""

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
from new_code.price_spillover_simulations import import_data, run_simulation

# Persistence setup
CONFIG_DIR = Path(__file__).parent.parent / ".dashboard_config"
CONFIG_FILE = CONFIG_DIR / "dashboard_state.json"
DEFAULTS_FILE = CONFIG_DIR / "dashboard_defaults.json"

def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None

def load_defaults():
    if DEFAULTS_FILE.exists():
        try:
            with open(DEFAULTS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None

def save_config(orderbooks, currency, frequency, lambda_target, lambda_value, lambda_up, lambda_down):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "orderbooks": orderbooks,
        "currency": currency,
        "frequency": frequency,
        "lambda_target": lambda_target,
        "lambda_value": lambda_value,
        "lambda_up": lambda_up,
        "lambda_down": lambda_down,
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def save_as_defaults(orderbooks, currency, frequency, lambda_target, lambda_value, lambda_up, lambda_down):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    defaults = {
        "orderbooks": orderbooks,
        "currency": currency,
        "frequency": frequency,
        "lambda_target": lambda_target,
        "lambda_value": lambda_value,
        "lambda_up": lambda_up,
        "lambda_down": lambda_down,
    }
    with open(DEFAULTS_FILE, 'w') as f:
        json.dump(defaults, f, indent=2)

st.set_page_config(layout="wide", page_title="Test Bench Dashboard")

st.title("Test Bench Simulation Dashboard")

# Load persisted config and defaults
saved_config = load_config()
saved_defaults = load_defaults()

# Initialize session state for custom orderbooks
if "custom_orderbooks" not in st.session_state:
    st.session_state.custom_orderbooks = {}

# Predefined orderbooks
predefined_orderbooks = {
    "no_orderbook": None,
    "Deep Narrow Tight": {"depth": 50_000_000, "width": 1, "spread": 0.005},
    "Deep Narrow Broad": {"depth": 50_000_000, "width": 1, "spread": 0.05},
    "Deep Wide Tight": {"depth": 50_000_000, "width": 10, "spread": 0.005},
    "Shallow Narrow Tight": {"depth": 5_000_000, "width": 1, "spread": 0.005},
    "Asymmetrical Depth": {"depth": (30_000_000, 50_000_000), "width": 1, "spread": 0.005},
}

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    # Currency and Frequency selection
    st.subheader("Simulation Parameters")
    default_currency = saved_config.get("currency") if saved_config else (saved_defaults.get("currency") if saved_defaults else "btc")
    default_frequency = saved_config.get("frequency") if saved_config else (saved_defaults.get("frequency") if saved_defaults else "30s")

    currency = st.radio(
        "Currency",
        ["btc", "sushi"],
        horizontal=True,
        index=0 if default_currency == "btc" else 1
    )
    frequency = st.radio(
        "Frequency",
        ["15s", "30s", "1min"],
        horizontal=True,
        index=["15s", "30s", "1min"].index(default_frequency)
    )
    show_hover = st.checkbox("Show hover info", value=False)

    # Rebalancing parameters
    st.subheader("Rebalancing Strategy")
    default_lambda_target = (saved_config.get("lambda_target") if saved_config else None) or (saved_defaults.get("lambda_target") if saved_defaults else False)
    default_lambda_value = (saved_config.get("lambda_value") if saved_config else None) or (saved_defaults.get("lambda_value") if saved_defaults else 1.5)
    default_lambda_up = (saved_config.get("lambda_up") if saved_config else None) or (saved_defaults.get("lambda_up") if saved_defaults else 4.0)
    default_lambda_down = (saved_config.get("lambda_down") if saved_config else None) or (saved_defaults.get("lambda_down") if saved_defaults else 1.25)

    lambda_target = st.checkbox("Use target rebalancing", value=default_lambda_target)
    if lambda_target:
        lambda_value = st.number_input("Lambda target", value=default_lambda_value, min_value=0.0, step=0.1)
    else:
        lambda_value = None

    lambda_up = st.slider("Lambda upper boundary", min_value=1.0, max_value=10.0, value=default_lambda_up, step=0.25)
    lambda_down = st.slider("Lambda lower boundary", min_value=0.1, max_value=5.0, value=default_lambda_down, step=0.1)

    # Orderbook selection
    st.subheader("Orderbook Selection")
    include_baseline = st.checkbox("Include baseline (no orderbook)", value=True)

    # Initialize orderbooks list if not exists
    if "orderbooks_list" not in st.session_state:
        if saved_config and "orderbooks" in saved_config:
            st.session_state.orderbooks_list = saved_config["orderbooks"]
        elif saved_defaults and "orderbooks" in saved_defaults:
            st.session_state.orderbooks_list = saved_defaults["orderbooks"]
        else:
            st.session_state.orderbooks_list = [
                {"_id": str(uuid.uuid4()), "name": "Deep Narrow Tight", "depth": 50_000_000, "width": 1.0, "spread": 0.005},
                {"_id": str(uuid.uuid4()), "name": "Deep Narrow Broad", "depth": 50_000_000, "width": 1.0, "spread": 0.05},
                {"_id": str(uuid.uuid4()), "name": "Deep Wide Tight", "depth": 50_000_000, "width": 10.0, "spread": 0.005},
                {"_id": str(uuid.uuid4()), "name": "Shallow Narrow Tight", "depth": 5_000_000, "width": 1.0, "spread": 0.005},
            ]

        # Ensure all loaded orderbooks have _id (for backwards compatibility)
        for ob in st.session_state.orderbooks_list:
            if "_id" not in ob:
                ob["_id"] = str(uuid.uuid4())

    # Display orderbook rows
    for idx, orderbook in enumerate(st.session_state.orderbooks_list):
        ob_id = orderbook.get("_id", str(uuid.uuid4()))
        if "_id" not in orderbook:
            orderbook["_id"] = ob_id

        # Name row with delete button
        col1, col2 = st.columns([6, 1])
        with col1:
            orderbook["name"] = st.text_input(
                "Name",
                value=orderbook["name"],
                key=f"name_{ob_id}",
                label_visibility="collapsed",
                placeholder="Orderbook name"
            )
        with col2:
            if st.button("🗑️", key=f"delete_{ob_id}", help="Delete this orderbook"):
                st.session_state.orderbooks_list = [ob for ob in st.session_state.orderbooks_list if ob.get("_id") != ob_id]
                st.rerun()

        # Parameters row (Depth, Width, Spread)
        param_col1, param_col2, param_col3 = st.columns([2, 1.5, 1.5])

        with param_col1:
            st.markdown("**Depth (M USD)**")
            # Convert list back to tuple if needed (from JSON deserialization)
            if isinstance(orderbook["depth"], list):
                orderbook["depth"] = tuple(orderbook["depth"])

            # Format depth for display in millions (handle both int and tuple)
            if isinstance(orderbook["depth"], tuple):
                depth_str = f"{orderbook['depth'][0] / 1_000_000:.1f}, {orderbook['depth'][1] / 1_000_000:.1f}"
            else:
                depth_str = f"{orderbook['depth'] / 1_000_000:.1f}"

            depth_input = st.text_input(
                "Depth",
                value=depth_str,
                key=f"depth_{ob_id}",
                label_visibility="collapsed",
                placeholder="e.g., 50 or 30, 50"
            )

            # Parse depth input (handle both single values and tuples), convert from millions
            try:
                if "," in depth_input:
                    parts = [int(float(p.strip()) * 1_000_000) for p in depth_input.split(",")]
                    orderbook["depth"] = tuple(parts) if len(parts) == 2 else parts[0]
                else:
                    orderbook["depth"] = int(float(depth_input) * 1_000_000)
            except ValueError:
                pass  # Keep previous value if parsing fails

        with param_col2:
            st.markdown("**Width (%)**")
            # Convert list back to tuple if needed (from JSON deserialization)
            if isinstance(orderbook["width"], list):
                orderbook["width"] = tuple(orderbook["width"])

            # Format width for display (handle both float and tuple)
            if isinstance(orderbook["width"], tuple):
                width_str = f"{orderbook['width'][0]}, {orderbook['width'][1]}"
            else:
                width_str = str(orderbook["width"])

            width_input = st.text_input(
                "Width",
                value=width_str,
                key=f"width_{ob_id}",
                label_visibility="collapsed",
                placeholder="e.g., 1 or 0.5, 1"
            )
            st.caption("e.g., 1 = 1%, 10 = 10%")

            # Parse width input (handle both single values and tuples)
            try:
                if "," in width_input:
                    parts = [float(p.strip()) for p in width_input.split(",")]
                    orderbook["width"] = tuple(parts) if len(parts) == 2 else parts[0]
                else:
                    orderbook["width"] = float(width_input)
            except ValueError:
                pass  # Keep previous value if parsing fails

        with param_col3:
            st.markdown("**Spread (%)**")
            # Convert list back to tuple if needed (from JSON deserialization)
            if isinstance(orderbook["spread"], list):
                orderbook["spread"] = tuple(orderbook["spread"])

            # Format spread for display (handle both float and tuple)
            if isinstance(orderbook["spread"], tuple):
                spread_str = f"{orderbook['spread'][0]}, {orderbook['spread'][1]}"
            else:
                spread_str = str(orderbook["spread"])

            spread_input = st.text_input(
                "Spread",
                value=spread_str,
                key=f"spread_{ob_id}",
                label_visibility="collapsed",
                placeholder="e.g., 0.005 or 0.003, 0.005"
            )
            st.caption("e.g., 0.005 = 0.5 bps, 0.05 = 5 bps")

            # Parse spread input (handle both single values and tuples)
            try:
                if "," in spread_input:
                    parts = [float(p.strip()) for p in spread_input.split(",")]
                    orderbook["spread"] = tuple(parts) if len(parts) == 2 else parts[0]
                else:
                    orderbook["spread"] = float(spread_input)
            except ValueError:
                pass  # Keep previous value if parsing fails

        st.divider()

    # Add row button
    if len(st.session_state.orderbooks_list) < 8:
        if st.button("+ Add Orderbook", type="primary"):
            st.session_state.orderbooks_list.append({
                "_id": str(uuid.uuid4()),
                "name": f"Orderbook {len(st.session_state.orderbooks_list) + 1}",
                "depth": 50_000_000,
                "width": 1.0,
                "spread": 0.005
            })
            st.rerun()

    # Settings management buttons
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save as Default", width='stretch'):
            save_as_defaults(st.session_state.orderbooks_list, currency, frequency, lambda_target, lambda_value, lambda_up, lambda_down)
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

    for orderbook in st.session_state.orderbooks_list:
        if orderbook["name"]:  # Only include if name is not empty
            selected_orderbooks[orderbook["name"]] = {
                "depth": orderbook["depth"],
                "width": orderbook["width"],
                "spread": orderbook["spread"]
            }

    # Save configuration
    save_config(st.session_state.orderbooks_list, currency, frequency, lambda_target, lambda_value, lambda_up, lambda_down)

# Main content
if not selected_orderbooks:
    st.warning("Please select at least one orderbook to run the simulation.")
else:
    # Run simulation button
    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            try:
                # Import data
                binance_data = import_data(currency, frequency)
                results = {}

                # Run simulations
                st.info("Orderbook Parameters:")
                for orderbook_name, orderbook in selected_orderbooks.items():
                    if orderbook is not None:
                        depth = orderbook.get('depth')
                        width = orderbook.get('width')
                        spread = orderbook.get('spread')
                        # Handle tuples
                        if isinstance(depth, (list, tuple)):
                            depth_display = f"({depth[0]:,}, {depth[1]:,})"
                        else:
                            depth_display = f"{depth:,}"
                        if isinstance(width, (list, tuple)):
                            width_display = f"({width[0]}, {width[1]})"
                        else:
                            width_display = f"{width}"
                        if isinstance(spread, (list, tuple)):
                            spread_display = f"({spread[0]}, {spread[1]})"
                        else:
                            spread_display = f"{spread}"

                        st.write(f"**{orderbook_name}** → Depth: {depth_display}, Width: {width_display}, Spread: {spread_display}")

                    result = run_simulation(
                        binance_data,
                        lambda_target=lambda_value,
                        lambda_up=lambda_up,
                        lambda_down=lambda_down,
                        orderbook=orderbook
                    )
                    results[orderbook_name] = result

                # Plot simulations
                fig = go.Figure()
                x = binance_data['timestamp'].values
                market_price = binance_data['price'].values

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
                        hovertemplate='<b>%{fullData.name}</b><br>Time: %{x|%H:%M:%S}<br>Price: $%{y:.2f}<extra></extra>' if show_hover else None,
                        hoverinfo='skip' if not show_hover else None,
                    ))

                # Plot actual market price
                fig.add_trace(go.Scatter(
                    x=x, y=market_price,
                    mode='lines',
                    name='Actual Market Price',
                    line=dict(color='black', width=2),
                    hovertemplate='<b>Actual Market Price</b><br>Time: %{x|%H:%M:%S}<br>Price: $%{y:.2f}<extra></extra>' if show_hover else None,
                    hoverinfo='skip' if not show_hover else None,
                ))

                # Format layout
                title_freq = frequency.replace('min', 'min ').upper()
                fig.update_layout(
                    title=f'{currency.upper()} {title_freq} Simulations Comparison',
                    xaxis_title='Timestamp',
                    yaxis_title=f'{currency.upper()} Price (USDT)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=800,
                    font=dict(size=12, color='black'),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    legend=dict(x=1.02, y=1, bgcolor='rgba(255, 255, 255, 0.9)', bordercolor='black', borderwidth=1, xanchor='left', yanchor='top', font=dict(color='black', size=12)),
                    title_font_color='black'
                )

                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_font_color='black', tickfont_color='black')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_font_color='black', tickfont_color='black')

                # Display results
                st.plotly_chart(fig, width='stretch')

                st.success("Simulation completed!")

            except Exception as e:
                st.error(f"Error running simulation: {str(e)}")
