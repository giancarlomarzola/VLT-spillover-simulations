import numpy as np
import matplotlib.pyplot as plt


def orderbook_depth(x, D, W, S):
    slope = D / (W - S)
    return np.where(
        x <= -W, D,
        np.where(x < -S, slope * (-x - S),
        np.where(x <= S, 0,
        np.where(x < W, slope * (x - S), D))))


D_vals = [1_000_000, 50_000_000] # in USD
W_vals = [1, 10] # in percent
S_vals = [0.005, 0.5] # in percent

combinations = [
    (D, W, S)
    for D in D_vals
    for W in W_vals
    for S in S_vals
]

n = len(combinations)
ncols = 2
nrows = -(-n // ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 12.5))
axes = axes.flatten()

x = np.linspace(-20, 20, 1000)

for ax, (D, W, S) in zip(axes, combinations):
    y = orderbook_depth(x, D, W, S)
    ax.fill_between(x, y, where=(x < 0), color="green", alpha=0.3, label="Cumulative Bid")
    ax.fill_between(x, y, where=(x > 0), color="red", alpha=0.3, label="Cumulative Ask")
    ax.plot(x[x <= 0], y[x <= 0], color="green", lw=0.7)
    ax.plot(x[x >= 0], y[x >= 0], color="red", lw=0.7)
    ax.set_title(f"Depth=${D/1_000_000:,}M, Width={W}%, Spread={S}%")
    ax.set_xlabel("Price deviation (%)")
    ax.set_xlim(-1.2*max(W_vals), 1.2*max(W_vals))
    ax.set_ylabel("Depth (USD)")
    #ax.set_ylim(0, max(D_vals)*1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(True, alpha=0.3)

fig.suptitle("Simulated Orderbook Archetypes", fontsize=14, y=0.98)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95), fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.94])

plt.savefig("Figures/orderbook_archetypes.png", dpi=300, bbox_inches="tight")
plt.show()