import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
from new_code.price_spillover_simulations import import_data, run_simulation

"""
TO RUN THE DASHBOARD: 
streamlit run new_code/dashboard.py
"""

st.set_page_config(layout="wide", page_title="Test Bench Dashboard")

st.title("Test Bench Simulation Dashboard")

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
    currency = st.radio("Currency", ["btc", "sushi"], horizontal=True)
    frequency = st.radio("Frequency", ["15s", "30s", "1min"], horizontal=True, index=1)
    show_hover = st.checkbox("Show hover info", value=False)

    # Rebalancing parameters
    st.subheader("Rebalancing Strategy")
    lambda_target = st.checkbox("Use target rebalancing", value=False)
    if lambda_target:
        lambda_value = st.number_input("Lambda target", value=1.5, min_value=0.0, step=0.1)
    else:
        lambda_value = None

    lambda_up = st.slider("Lambda upper boundary", min_value=1.0, max_value=10.0, value=4.0, step=0.25)
    lambda_down = st.slider("Lambda lower boundary", min_value=0.1, max_value=5.0, value=1.25, step=0.1)

    # Orderbook selection
    st.subheader("Orderbook Selection")

    # Initialize orderbooks list if not exists
    if "orderbooks_list" not in st.session_state:
        st.session_state.orderbooks_list = [
            {"name": "Deep Narrow Tight", "depth": 50_000_000, "width": 1.0, "spread": 0.005},
            {"name": "Deep Narrow Broad", "depth": 50_000_000, "width": 1.0, "spread": 0.05},
            {"name": "Deep Wide Tight", "depth": 50_000_000, "width": 10.0, "spread": 0.005},
            {"name": "Shallow Narrow Tight", "depth": 5_000_000, "width": 1.0, "spread": 0.005},
        ]

    # Display header with column titles
    col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1, 1, 0.5])
    with col1:
        st.markdown("**Name**")
    with col2:
        st.markdown("**Depth (USDT)**")
    with col3:
        st.markdown("**Width (%)**")
    with col4:
        st.markdown("**Spread (%)**")
    with col5:
        st.markdown("")

    # Display orderbook rows
    for idx in range(len(st.session_state.orderbooks_list)):
        orderbook = st.session_state.orderbooks_list[idx]
        col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1, 1, 0.5])

        with col1:
            orderbook["name"] = st.text_input(
                "Name",
                value=orderbook["name"],
                key=f"name_{idx}",
                label_visibility="collapsed",
                placeholder="Orderbook name"
            )

        with col2:
            # Format depth for display (handle both int and tuple)
            if isinstance(orderbook["depth"], tuple):
                depth_str = f"{orderbook['depth'][0]:,}, {orderbook['depth'][1]:,}"
            else:
                depth_str = f"{orderbook['depth']:,}"

            depth_input = st.text_input(
                "Depth",
                value=depth_str,
                key=f"depth_{idx}",
                label_visibility="collapsed",
                placeholder="e.g., 50000000 or 30000000, 50000000"
            )

            # Parse depth input (handle both single values and tuples)
            try:
                if "," in depth_input:
                    parts = [int(p.strip().replace(",", "")) for p in depth_input.split(",")]
                    orderbook["depth"] = tuple(parts) if len(parts) == 2 else parts[0]
                else:
                    orderbook["depth"] = int(depth_input.replace(",", ""))
            except ValueError:
                pass  # Keep previous value if parsing fails

        with col3:
            orderbook["width"] = st.number_input(
                "Width",
                value=orderbook["width"],
                min_value=0.1,
                max_value=50.0,
                step=0.1,
                key=f"width_{idx}",
                label_visibility="collapsed"
            )

        with col4:
            orderbook["spread"] = st.number_input(
                "Spread",
                value=orderbook["spread"],
                min_value=0.001,
                max_value=1.0,
                step=0.001,
                key=f"spread_{idx}",
                label_visibility="collapsed"
            )

        with col5:
            if st.button("🗑️", key=f"delete_{idx}", help="Delete this orderbook"):
                st.session_state.orderbooks_list.pop(idx)
                st.rerun()

    # Add row button
    if len(st.session_state.orderbooks_list) < 8:
        if st.button("+ Add Orderbook", type="primary"):
            st.session_state.orderbooks_list.append({
                "name": f"Orderbook {len(st.session_state.orderbooks_list) + 1}",
                "depth": 50_000_000,
                "width": 1.0,
                "spread": 0.005
            })
            st.rerun()

    # Build selected_orderbooks from the list
    selected_orderbooks = {}
    for orderbook in st.session_state.orderbooks_list:
        if orderbook["name"]:  # Only include if name is not empty
            selected_orderbooks[orderbook["name"]] = {
                "depth": orderbook["depth"],
                "width": orderbook["width"],
                "spread": orderbook["spread"]
            }

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
                for orderbook_name, orderbook in selected_orderbooks.items():
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
                st.plotly_chart(fig, use_container_width=True)

                st.success("Simulation completed!")

            except Exception as e:
                st.error(f"Error running simulation: {str(e)}")
