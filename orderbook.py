# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:51:03 2023

@author: gm399
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def make_orderbook(
    alpha, beta, worst_ask, worst_bid, ask_depth, bid_depth, max_deviation, tick_size
):

    # Get overall number of ticks - this is the length of the df
    total_ticks = int(max_deviation / tick_size)

    # Find how many prices have an existing bid/ask
    bid_ticks = round((worst_bid - beta) / tick_size)
    ask_ticks = round((worst_ask - alpha) / tick_size)

    # Find how many ticks to skip at beginning, because they belong to spread
    beta_ticks = round(beta / tick_size)
    alpha_ticks = round(alpha / tick_size)

    # Find necessary size of bid/ask to have constant size at all prices
    bid_size = bid_depth / bid_ticks
    ask_size = ask_depth / ask_ticks

    # Setting up df with prices and all sizes = 0
    orderbook = pd.DataFrame(
        {
            "price": np.linspace(0, max_deviation, total_ticks),
            "bid_size": np.zeros(total_ticks),
            "ask_size": np.zeros(total_ticks),
        }
    )

    # Replace the sizes for those prices where bids/asks are present
    orderbook.loc[beta_ticks + 1 : bid_ticks + beta_ticks, "bid_size"] = bid_size
    orderbook.loc[alpha_ticks + 1 : ask_ticks + alpha_ticks, "ask_size"] = ask_size

    # Add cumulative size columns
    orderbook["cumulative_bid"] = orderbook["bid_size"].cumsum()
    orderbook["cumulative_ask"] = orderbook["ask_size"].cumsum()

    return orderbook


def plot_orderbook(orderbook, plot_path=None, plot_name=None):
    # setting x's and y's for the plot
    x_bid = orderbook["price"].multiply(-1)
    y_bid = orderbook["cumulative_bid"]

    x_ask = orderbook["price"]
    y_ask = orderbook["cumulative_ask"]

    # Create plot
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 7), dpi=300)

    # Plot the bids in green
    plt.plot(
        x_bid,
        y_bid,
        color="green",
        label="Cumulative Bid",
        drawstyle="steps",
        linewidth=0.8,
    )
    # Plot the asks in red
    plt.plot(
        x_ask,
        y_ask,
        color="red",
        label="Cumulative Ask",
        drawstyle="steps",
        linewidth=0.8,
    )

    # Fill area under bids and asks with colour
    plt.fill_between(x_bid, y_bid, step="pre", alpha=0.3, color="green")
    plt.fill_between(x_ask, y_ask, step="pre", alpha=0.3, color="red")

    # format x-axis
    max_price = orderbook["price"].max()
    plt.xlabel("Price (% deviation from mid price)", fontsize=20)
    plt.xticks(np.arange(-max_price, max_price * 1.2, 2))  # , rotation=70)
    plt.xticks(np.arange(-max_price, max_price * 1.2, 1), minor=True)
    plt.xlim(-max_price, max_price)

    # format y-axis
    plt.ylabel("Orderbook Depth (Tokens)", fontsize=20)
    max_depth = max(
        orderbook["cumulative_ask"].iloc[-1], orderbook["cumulative_bid"].iloc[-1]
    )

    if max_depth >= 1_000_000:  # SUSHI deep
        y_tick_step = 250_000
    elif max_depth >= 200_000:  # SUSHI Shallow
        y_tick_step = 50_000
    elif max_depth >= 8000:  # ETH deep
        y_tick_step = 1000
    elif max_depth >= 1500:  # ETH shallow
        y_tick_step = 250
    elif max_depth >= 500:
        y_tick_step = 100
    else:
        y_tick_step = 20

    plt.yticks(np.arange(0, max_depth * 1.3, y_tick_step))
    axes.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    plt.ylim(0, max_depth * 1.3)

    # set tick fontsizes
    axes.tick_params(axis="both", which="major", labelsize=18)

    # Add a legend and grid
    plt.legend(loc="upper left", fontsize=18)
    plt.grid()

    fig = fig.get_figure()
    # Save the plot
    if plot_path != None:
        fig.savefig(
            f"{plot_path}/{plot_name}_orderbook.png", bbox_inches="tight"
        )  # , dpi=300)

    # Show the plot
    return fig
    plt.show()
