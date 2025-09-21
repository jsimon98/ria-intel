# scripts/build_gold.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from riaintel.etl.etl_utils import load_silver


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = Path(os.getenv("RIA_RUNTIME_DIR", PROJECT_ROOT / "runtime")).resolve()
SILVER_DIR = Path(os.getenv("RIA_SILVER_DIR", RUNTIME_DIR / "silver")).resolve()
GOLD_DIR = Path(os.getenv("RIA_GOLD_DIR", RUNTIME_DIR / "gold")).resolve()

TRUE_VALUES = {"Y", "YES", "TRUE", "T", "1"}


def prepare_silver(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns for ordering and filtering the SILVER snapshot."""

    if df.empty:
        return pd.DataFrame(columns=["crd", "report_date", "date_submitted", "filing_id"])

    if "1E1" not in df.columns:
        raise KeyError("Missing CRD column '1E1' in SILVER data")
    if "REPORT_DATE" not in df.columns:
        raise KeyError("Missing 'REPORT_DATE' in SILVER data")

    working = df.copy()

    crd_numeric = pd.to_numeric(working["1E1"], errors="coerce")
    report_date = pd.to_datetime(working["REPORT_DATE"], format="%Y%m%d", errors="coerce")
    date_submitted = pd.to_datetime(working.get("DATESUBMITTED"), errors="coerce")
    filing_id = working.get("FILING_ID")
    if filing_id is None:
        filing_id = pd.Series(pd.NA, index=working.index, dtype="string")
    else:
        filing_id = filing_id.astype("string")

    working = working.assign(
        crd=crd_numeric,
        report_date=report_date.dt.normalize(),
        date_submitted=date_submitted,
        filing_id=filing_id,
    )

    mask = working["crd"].notna() & working["report_date"].notna()
    working = working.loc[mask].copy()
    working["crd"] = working["crd"].astype("Int64")

    working = working.sort_values(
        ["crd", "report_date", "date_submitted", "filing_id"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return working


def _latest_per_period(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    ordered = df.sort_values(
        ["crd", "report_date", "date_submitted", "filing_id"],
        ascending=[True, False, False, False],
        na_position="last",
    )
    latest = ordered.drop_duplicates(subset=["crd", "report_date"], keep="first")
    return latest.sort_values(["crd", "report_date"]).reset_index(drop=True)


def _latest_per_crd(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    ordered = df.sort_values(
        ["crd", "report_date", "date_submitted", "filing_id"],
        ascending=[True, False, False, False],
        na_position="last",
    )
    latest = ordered.drop_duplicates(subset=["crd"], keep="first")
    return latest.reset_index(drop=True)


def _clean_text(series: pd.Series, length: int) -> pd.Series:
    if series is None:
        return pd.Series([None] * length, dtype="object")
    cleaned = series.astype("string").str.strip()
    return cleaned.where(cleaned.notna(), None)


def _boolean_from_series(series: pd.Series, length: int) -> pd.Series:
    if series is None:
        return pd.Series([False] * length, dtype=bool)
    normalized = series.fillna("").astype(str).str.strip().str.upper()
    return normalized.isin(TRUE_VALUES).astype(bool)


def _numeric_float(series: pd.Series, length: int) -> pd.Series:
    if series is None:
        return pd.Series([pd.NA] * length, dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def _numeric_int(series: pd.Series, length: int) -> pd.Series:
    if series is None:
        return pd.Series([pd.NA] * length, dtype="Int64")
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.round().astype("Int64")


def build_firm_master(df: pd.DataFrame) -> pd.DataFrame:
    latest = _latest_per_crd(df)
    if latest.empty:
        columns = [
            "CRD Number",
            "Firm Legal Name",
            "Primary Business Name",
            "Ownership Type",
            "Custody",
            "Disciplinary Info Provided",
            "Employees – Total",
            "Regulatory AUM – Total (USD)",
            "Regulatory AUM – Non-Discretionary (USD)",
        ]
        return pd.DataFrame(columns=columns)

    length = len(latest)

    legal = _clean_text(latest.get("1A"), length)
    business = _clean_text(latest.get("1B1"), length)
    ownership = _clean_text(latest.get("3A"), length)
    custody = _boolean_from_series(latest.get("7B"), length)
    discipline = _boolean_from_series(latest.get("11"), length)
    employees = _numeric_int(latest.get("5A"), length)
    aum_total = _numeric_float(latest.get("5F3"), length)
    aum_nondisc = _numeric_float(latest.get("5F2B"), length)

    master = pd.DataFrame(
        {
            "CRD Number": latest["crd"].astype("Int64"),
            "Firm Legal Name": legal,
            "Primary Business Name": business,
            "Ownership Type": ownership,
            "Custody": custody,
            "Disciplinary Info Provided": discipline,
            "Employees – Total": employees,
            "Regulatory AUM – Total (USD)": aum_total,
            "Regulatory AUM – Non-Discretionary (USD)": aum_nondisc,
        }
    )

    return master


def build_firm_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    per_period = _latest_per_period(df)
    if per_period.empty:
        columns = [
            "CRD Number",
            "Report Date",
            "Filing ID",
            "Regulatory AUM – Discretionary (USD)",
            "Regulatory AUM – Non-Discretionary (USD)",
            "Regulatory AUM – Total (USD)",
            "Employees – Total",
        ]
        return pd.DataFrame(columns=columns)

    length = len(per_period)

    aum_disc = _numeric_float(per_period.get("5F2A"), length)
    aum_nondisc = _numeric_float(per_period.get("5F2B"), length)
    aum_total = _numeric_float(per_period.get("5F3"), length)
    employees = _numeric_int(per_period.get("5A"), length)
    filing_id = per_period["filing_id"].astype("string")
    filing_id = filing_id.where(filing_id.notna(), None)

    timeseries = pd.DataFrame(
        {
            "CRD Number": per_period["crd"].astype("Int64"),
            "Report Date": per_period["report_date"],
            "Filing ID": filing_id,
            "Regulatory AUM – Discretionary (USD)": aum_disc,
            "Regulatory AUM – Non-Discretionary (USD)": aum_nondisc,
            "Regulatory AUM – Total (USD)": aum_total,
            "Employees – Total": employees,
        }
    )

    return timeseries.sort_values(["CRD Number", "Report Date"]).reset_index(drop=True)


def _prepare_notice_components(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    per_period = _latest_per_period(df)
    state_cols = sorted([c for c in per_period.columns if c.startswith("2_")])
    if not state_cols:
        empty_bool = pd.DataFrame(index=per_period.index)
        return per_period, empty_bool, []

    length = len(per_period)
    state_codes = [c.split("_", 1)[1] for c in state_cols]
    matrix = {
        code: _boolean_from_series(per_period[col], length)
        for col, code in zip(state_cols, state_codes)
    }
    state_bool = pd.DataFrame(matrix, index=per_period.index)
    return per_period, state_bool, state_codes


def build_notice_filings_wide(df: pd.DataFrame) -> pd.DataFrame:
    per_period, state_bool, state_codes = _prepare_notice_components(df)
    if per_period.empty:
        base_columns = ["CRD Number", "Report Date", "Filing ID", "States Filed", "States Count"]
        return pd.DataFrame(columns=base_columns)

    filing_id = per_period["filing_id"].astype("string")
    filing_id = filing_id.where(filing_id.notna(), None)

    if state_bool.empty:
        states_count = pd.Series([0] * len(per_period), dtype="Int64")
        states_filed = pd.Series([None] * len(per_period), dtype="object")
        columns = {}
    else:
        states_count = state_bool.sum(axis=1).astype("Int64")
        states_list = state_bool.apply(lambda row: [code for code, flag in row.items() if flag], axis=1)
        states_filed = states_list.apply(lambda codes: "|".join(codes) if codes else None)
        columns = {code: state_bool[code] for code in state_codes}

    wide = pd.DataFrame(
        {
            "CRD Number": per_period["crd"].astype("Int64"),
            "Report Date": per_period["report_date"],
            "Filing ID": filing_id,
            "States Filed": states_filed,
            "States Count": states_count,
        }
    )

    for code, series in columns.items():
        wide[code] = series.astype(bool)

    return wide.sort_values(["CRD Number", "Report Date"]).reset_index(drop=True)


def build_notice_filings_long(df: pd.DataFrame) -> pd.DataFrame:
    per_period, state_bool, state_codes = _prepare_notice_components(df)
    if per_period.empty or state_bool.empty:
        columns = ["CRD Number", "Report Date", "State"]
        return pd.DataFrame(columns=columns)

    long_base = per_period[["crd", "report_date"]].join(state_bool)
    melted = long_base.melt(id_vars=["crd", "report_date"], var_name="State", value_name="Filed")
    melted = melted[melted["Filed"]].drop(columns=["Filed"]).reset_index(drop=True)
    melted.rename(columns={"crd": "CRD Number", "report_date": "Report Date"}, inplace=True)
    melted["CRD Number"] = melted["CRD Number"].astype("Int64")
    return melted.sort_values(["CRD Number", "Report Date", "State"]).reset_index(drop=True)


def build_firms_latest(firm_master: pd.DataFrame, firm_timeseries: pd.DataFrame) -> pd.DataFrame:
    if firm_master.empty or firm_timeseries.empty:
        return pd.DataFrame(columns=[
            "CRD Number",
            "Firm Legal Name",
            "Primary Business Name",
            "Ownership Type",
            "Custody",
            "Disciplinary Info Provided",
            "Report Date",
            "Filing ID",
            "Regulatory AUM – Discretionary (USD)",
            "Regulatory AUM – Non-Discretionary (USD)",
            "Regulatory AUM – Total (USD)",
            "Employees – Total",
        ])

    latest_metrics = firm_timeseries.sort_values(["CRD Number", "Report Date"], ascending=[True, False])
    latest_metrics = latest_metrics.drop_duplicates(subset=["CRD Number"], keep="first")

    static_cols = [
        "CRD Number",
        "Firm Legal Name",
        "Primary Business Name",
        "Ownership Type",
        "Custody",
        "Disciplinary Info Provided",
    ]
    static = firm_master[static_cols]

    merged = static.merge(latest_metrics, on="CRD Number", how="inner")
    return merged.sort_values("CRD Number").reset_index(drop=True)


def build_notice_state_counts(notice_wide: pd.DataFrame) -> pd.DataFrame:
    if notice_wide.empty:
        return pd.DataFrame(columns=["State", "Firm Count"])

    state_cols = [c for c in notice_wide.columns if notice_wide[c].dtype == bool]
    if not state_cols:
        return pd.DataFrame(columns=["State", "Firm Count"])

    latest = notice_wide.sort_values(["CRD Number", "Report Date"], ascending=[True, False])
    latest = latest.drop_duplicates(subset=["CRD Number"], keep="first")

    counts = latest[state_cols].sum(axis=0)
    counts = counts.astype("Int64")
    summary = counts.reset_index()
    summary.columns = ["State", "Firm Count"]
    return summary.sort_values(["Firm Count", "State"], ascending=[False, True]).reset_index(drop=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:  # pragma: no cover - helpful runtime error
        raise RuntimeError(
            f"Writing {path.name} requires a parquet engine (pyarrow or fastparquet)."
        ) from exc


def main() -> None:
    print("=== Building GOLD tables ===")
    print(f"SILVER_DIR: {SILVER_DIR}")
    print(f"GOLD_DIR:   {GOLD_DIR}")

    print("\n[1/6] Loading SILVER data...")
    silver = load_silver(SILVER_DIR)
    print(f"  SILVER shape: {silver.shape}")

    print("\n[2/6] Preparing keys and ordering...")
    prepared = prepare_silver(silver)
    print(f"  Prepared shape: {prepared.shape}")

    print("\n[3/6] Building firm_master...")
    firm_master = build_firm_master(prepared)
    print(f"  firm_master shape: {firm_master.shape}")
    save_parquet(firm_master, GOLD_DIR / "firm_master.parquet")

    print("\n[4/6] Building firm_timeseries...")
    firm_timeseries = build_firm_timeseries(prepared)
    print(f"  firm_timeseries shape: {firm_timeseries.shape}")
    save_parquet(firm_timeseries, GOLD_DIR / "firm_timeseries.parquet")

    print("\n[5/6] Building notice filings (wide & long)...")
    notice_wide = build_notice_filings_wide(prepared)
    print(f"  notice_filings_wide shape: {notice_wide.shape}")
    save_parquet(notice_wide, GOLD_DIR / "notice_filings_wide.parquet")

    notice_long = build_notice_filings_long(prepared)
    print(f"  notice_filings_long shape: {notice_long.shape}")
    save_parquet(notice_long, GOLD_DIR / "notice_filings_long.parquet")

    print("\n[6/6] Building helper views...")
    firms_latest = build_firms_latest(firm_master, firm_timeseries)
    print(f"  firms_latest shape: {firms_latest.shape}")
    save_parquet(firms_latest, GOLD_DIR / "firms_latest.parquet")

    notice_state_counts = build_notice_state_counts(notice_wide)
    print(f"  notice_state_counts shape: {notice_state_counts.shape}")
    save_parquet(notice_state_counts, GOLD_DIR / "notice_state_counts.parquet")

    print("\nDone. GOLD tables written to:", GOLD_DIR)


if __name__ == "__main__":
    main()

