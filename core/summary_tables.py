"""
Tables of averages and standard deviations for the runs saved by simulation_from_gbm.py.

Reads every per-seed file in <experiment_dir>/iterations/ and averages each (price series,
orderbook) cell across seeds, with a second table of the standard deviations across seeds.
It runs no simulations, so it can be called at any point to pick up newly saved iterations.

Runs that fell into a fatal feedback loop (see core.simulation.flag_feedback_loops) are
counted in n_feedback_loops but left out of every average: their price and NAV values run
off to the 1e-10 floor or to overflow, and would swamp the other runs.
"""

from pathlib import Path

import pandas as pd
from IPython.display import display
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from core.simulation import flag_feedback_loops

RATE_PREFIXES = ("wipeout_rate_", "fatality_rate_", "feedback_loop_rate")  # shares of runs, shown in %


def build_summary_tables(experiment_dir: Path, show: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the averages and standard deviations tables for an experiment and save them to
    <experiment_dir>/summary_table.xlsx ("summary" and "stdev" sheets).

    show: display both tables, formatted (bps, %, rounded times), as well as saving them.
    Returns (summary, stdev) with raw values.
    """
    experiment_dir = Path(experiment_dir)
    results_dir = experiment_dir / "iterations"  # one parquet file per seed
    paths = sorted(results_dir.glob("seed_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No saved iterations in {results_dir}; run simulation_from_gbm.py first")

    # One row per (seed, price series, orderbook) run
    runs = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    print(f"{runs['seed'].nunique()} iterations, seeds {runs['seed'].min()} to {runs['seed'].max()}")

    loops = flag_feedback_loops(runs)
    print(f"{loops['feedback_loop'].sum():,} of {len(runs):,} runs hit a fatal feedback loop, left out of the averages")

    # A wipeout time only exists for the runs that wiped out, so its average is over those runs
    # only. Next to it:
    #   wipeout_rate_<product>: share of the runs without a feedback loop where NAV hit zero.
    #   fatality_rate_<product>: share of all runs where NAV hit zero, feedback loops included.
    #       This is the overall risk of the product; wipeout_rate_ shows the risk that is left
    #       once loops are set aside.
    columns = []
    fatality_columns = {}  # wipeout_rate_ column -> its fatality_rate_ column
    for column in runs.columns.drop(["seed", "price_series", "orderbook"]):
        columns.append(column)
        if column.startswith("wipeout_time_"):
            rate_column = column.replace("wipeout_time_", "wipeout_rate_")
            runs[rate_column] = runs[column].notna()
            columns.append(rate_column)
            fatality_columns[rate_column] = column.replace("wipeout_time_", "fatality_rate_")

    # Average across seeds, over the runs without a feedback loop. Statistics that are NaN in a
    # run (e.g. no rebalances) are skipped. Remove entries from `columns` to drop them from the table.
    keys = ["price_series", "orderbook"]
    grouped = runs.groupby(keys, sort=False)
    summary = runs[~loops["feedback_loop"]].groupby(keys, sort=False)[columns].mean()
    summary = summary.reindex(grouped.size().index)  # keep cells where every run was a loop, as blanks

    # Rates over all runs, loops included
    for rate_column, fatality_column in fatality_columns.items():
        summary.insert(summary.columns.get_loc(rate_column) + 1, fatality_column, grouped[rate_column].mean())

    # Feedback loop counts per cell, split by direction (price run up / run down), and as a share of runs
    loop_counts = loops.groupby([runs[key] for key in keys], sort=False).sum()
    summary.insert(0, "n_iterations", grouped.size())
    summary.insert(1, "n_feedback_loops", loop_counts["feedback_loop"])
    summary.insert(2, "n_feedback_loops_up", loop_counts["feedback_loop_up"])
    summary.insert(3, "n_feedback_loops_down", loop_counts["feedback_loop_down"])
    summary.insert(4, "feedback_loop_rate", summary["n_feedback_loops"] / summary["n_iterations"])

    # Standard deviation across seeds of each averaged column, over the same runs as the averages
    # (no feedback loops, NaN skipped). Rates and counts of runs are left out: a rate is a share of
    # runs, so its spread follows from the rate itself. n_runs is the number of runs behind each
    # cell (for standard errors, std / sqrt(n)); a wipeout time's std is over the wiped-out runs only.
    std_columns = [column for column in columns if not column.startswith(RATE_PREFIXES)]
    stdev = runs[~loops["feedback_loop"]].groupby(keys, sort=False)[std_columns].std()
    stdev = stdev.reindex(grouped.size().index)
    stdev.insert(0, "n_runs", summary["n_iterations"] - summary["n_feedback_loops"])

    if show:
        formats = display_formats(summary.columns)
        display(summary.style.format(formats, na_rep="-"))
        stdev_formats = {column: formats[column] for column in stdev.columns if column in formats}
        display(stdev.style.format(stdev_formats, na_rep="-"))

    save_excel(summary, stdev, experiment_dir / "summary_table.xlsx")
    return summary, stdev


def display_formats(columns: pd.Index) -> dict:
    """Display formats: sizes, depths and NAVs in quote currency, price impacts and deviations in bps, vols in %."""
    formats = {column: "{:,.0f}" for column in columns}  # counts (n_iterations, n_feedback_loops...), sizes, NAVs
    formats |= {
        column: (lambda x: f"{x * 1e4:.2f} bps")
        for column in columns
        if column.endswith(("_impact", "_price_deviation"))
    }
    formats |= {column: (lambda x: str(x.round("s"))) for column in columns if column.startswith("wipeout_time_")}
    formats |= {column: "{:.1%}" for column in columns if column.startswith(RATE_PREFIXES)}
    formats |= {column: "{:,.1f}" for column in columns if column.startswith(("n_rebalances", "n_outside_band_"))}
    formats |= {
        "longest_rebalance_run": "{:,.1f}",
        "fill_ratio": "{:.2%}",
        "realised_vol_raw": "{:.2%}",
        "realised_vol_simulated": "{:.2%}",
        "realised_vol_ratio": "{:.5f}",
    }
    return formats


# Both tables are saved as Excel, overwriting the previous file: averages on the "summary"
# sheet, standard deviations on the "stdev" sheet. Values are the raw numbers (decimals rather
# than bps or %), so they can be used in calculations; wipeout times are in days, as Excel has
# no elapsed-time type.
def save_excel(summary: pd.DataFrame, stdev: pd.DataFrame, excel_path: Path) -> None:
    """Save both tables to excel_path, or print a warning if the file is open in Excel."""
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            _write_sheet(writer, _to_excel_values(summary), "summary")
            _write_sheet(writer, _to_excel_values(stdev), "stdev")
        print(f"Tables saved to {excel_path}")
    except PermissionError:
        print(f"Could not save {excel_path}: close it in Excel and run this again")


def _to_excel_values(table: pd.DataFrame) -> pd.DataFrame:
    """Copy of a table with wipeout times converted to days."""
    table = table.copy()
    for column in [column for column in table.columns if column.startswith("wipeout_time_")]:
        table[column] = table[column].dt.total_seconds() / 86_400
        table = table.rename(columns={column: f"{column}_days"})
    return table


def _excel_number_format(column: str) -> str:
    """Excel display format for a column. It changes only how the number is shown, not its value."""
    if column.endswith(("_impact", "_price_deviation")):
        return "0.000000"  # decimals, shown to 0.01 bps
    if column.startswith("wipeout_time_"):
        return "#,##0.00"  # days
    if column.startswith(RATE_PREFIXES):
        return "0.0%"
    if column.startswith(("n_rebalances", "n_outside_band_")) or column == "longest_rebalance_run":
        return "#,##0.0"
    if column in ("fill_ratio", "realised_vol_raw", "realised_vol_simulated"):
        return "0.00%"
    if column == "realised_vol_ratio":
        return "0.00000"
    return "#,##0"  # counts, sizes, depths and NAVs


def _write_sheet(writer: pd.ExcelWriter, table: pd.DataFrame, sheet_name: str) -> None:
    """Write a table to its own sheet, with a uniform font, borders, number formats and filters."""
    table.to_excel(writer, sheet_name=sheet_name, merge_cells=False)
    sheet = writer.sheets[sheet_name]
    n_index = table.index.nlevels
    font = Font(name="Calibri", size=11)
    edge = Side(style="thin", color="BFBFBF")
    border = Border(left=edge, right=edge, top=edge, bottom=edge)

    # Every cell gets the same font and border, so nothing is left on pandas' defaults
    for position, cells in enumerate(sheet.iter_cols(), start=1):
        header, body = cells[0], cells[1:]
        is_index = position <= n_index
        header.font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        header.fill = PatternFill("solid", fgColor="1F4E78")
        header.alignment = Alignment(horizontal="center", vertical="center")
        header.border = border
        number_format = "General" if is_index else _excel_number_format(header.value)
        for cell in body:
            cell.font = font
            cell.border = border
            cell.alignment = Alignment(horizontal="left" if is_index else "right", vertical="center")
            cell.number_format = number_format
        # Wide enough for the header (plus its filter button) and for the largest NAVs
        longest = max([len(str(header.value))] + [len(str(cell.value)) for cell in body if is_index])
        sheet.column_dimensions[get_column_letter(position)].width = max(longest + 5, 16)

    sheet.row_dimensions[1].height = 22
    sheet.freeze_panes = sheet.cell(row=2, column=n_index + 1)  # keep header and row labels in view
    sheet.auto_filter.ref = sheet.dimensions
