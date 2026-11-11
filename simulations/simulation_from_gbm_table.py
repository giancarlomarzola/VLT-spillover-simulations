"""
Table of averages for the runs saved by simulation_from_gbm.py.

Reads every per-seed file in data/outputs/<experiment>/iterations/ and averages each
(price series, orderbook) cell across seeds. It runs no simulations, so it can be re-run
at any point to pick up newly saved iterations.
"""

import pandas as pd
from IPython.display import display

from utils.paths import DATA_OUTPUTS

experiment = "simulation_from_gbm"  # folder written by simulation_from_gbm.py

experiment_dir = DATA_OUTPUTS / experiment
results_dir = experiment_dir / "iterations"  # one parquet file per seed
paths = sorted(results_dir.glob("seed_*.parquet"))
if not paths:
    raise FileNotFoundError(f"No saved iterations in {results_dir}; run simulation_from_gbm.py first")

# One row per (seed, price series, orderbook) run
runs = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
print(f"{runs['seed'].nunique()} iterations, seeds {runs['seed'].min()} to {runs['seed'].max()}")

# A wipeout time only exists for the runs that wiped out, so its average is over those runs
# only. The wipeout rate (share of iterations where NAV hit zero) is shown next to it.
columns = []
for column in runs.columns.drop(["seed", "price_series", "orderbook"]):
    columns.append(column)
    if column.startswith("wipeout_time_"):
        rate_column = column.replace("wipeout_time_", "wipeout_rate_")
        runs[rate_column] = runs[column].notna()
        columns.append(rate_column)

# Average across seeds. Statistics that are NaN in a run (e.g. no rebalances) are skipped.
# Remove entries from `columns` to drop them from the table.
grouped = runs.groupby(["price_series", "orderbook"], sort=False)
summary = grouped[columns].mean()
summary.insert(0, "n_iterations", grouped.size())

# Sizes, depths and NAVs in quote currency, price impacts and deviations in bps, vols in %
formats = {column: "{:,.0f}" for column in summary.columns}
formats |= {
    column: (lambda x: f"{x * 1e4:.2f} bps")
    for column in summary.columns
    if column.endswith(("_impact", "_price_deviation"))
}
formats |= {column: (lambda x: str(x.round("s"))) for column in summary.columns if column.startswith("wipeout_time_")}
formats |= {column: "{:.1%}" for column in summary.columns if column.startswith("wipeout_rate_")}
formats |= {column: "{:,.1f}" for column in summary.columns if column.startswith(("n_rebalances", "n_outside_band_"))}
formats |= {
    "longest_rebalance_run": "{:,.1f}",
    "fill_ratio": "{:.2%}",
    "realised_vol_raw": "{:.2%}",
    "realised_vol_simulated": "{:.2%}",
    "realised_vol_ratio": "{:.5f}",
}
display(summary.style.format(formats, na_rep="-"))

# Also save the table as Excel, overwriting the previous one. Values are the raw numbers
# (decimals rather than bps or %), so they can be used in calculations; wipeout times are
# in days, as Excel has no elapsed-time type.
excel_summary = summary.copy()
for column in [column for column in summary.columns if column.startswith("wipeout_time_")]:
    excel_summary[column] = summary[column].dt.total_seconds() / 86_400
    excel_summary = excel_summary.rename(columns={column: f"{column}_days"})
excel_path = experiment_dir / "summary_table.xlsx"
try:
    excel_summary.to_excel(excel_path, merge_cells=False)
    print(f"Table saved to {excel_path}")
except PermissionError:
    print(f"Could not save {excel_path}: close it in Excel and run this again")
