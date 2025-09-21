# scripts/build_gold.py
from pathlib import Path
import sys, os
import pandas as pd

from riaintel.etl.etl_utils import (
    SILVER_DIR, GOLD_DIR,
    load_silver, write_parquet,
    prepare_keys, build_timeseries, build_firm_master,
    apply_gold_schema, get_table_schema,    # no audit_columns printing here
    apply_bi_labels,                        # used for display + previews
)

SCHEMAS_PATH = Path(__file__).resolve().parent.parent / "config" / "schemas.yaml"
LABEL_YAML   = Path(__file__).resolve().parent.parent / "config" / "base_a_base_b_columns.yaml"
PREVIEW_DIR  = GOLD_DIR / "_bi_preview"   # labeled, human-friendly, non-canonical


def _size_mb(p: Path) -> str:
    try:
        return f"{os.stat(p).st_size/1024/1024:,.2f} MB"
    except Exception:
        return "n/a"


def _label_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with BI labels applied (if LABEL_YAML exists)."""
    try:
        return apply_bi_labels(df.copy(), str(LABEL_YAML)) if LABEL_YAML.exists() else df
    except Exception:
        # If label mapping fails for any reason, just return the original df
        return df


def _na_report(df: pd.DataFrame, title: str, limit: int = 15):
    """Print NA counts with BI-labeled column names for readability."""
    df_disp = _label_df_for_display(df)
    na = df_disp.isna().sum()
    na = na[na > 0].sort_values(ascending=False)
    print(f"  [NA] {title}: total_na_cells={int(na.sum())}")
    if na.empty:
        print("    - none"); return
    n = max(len(df_disp), 1)
    for c, v in na.head(limit).items():
        print(f"    - {c}: {int(v)} ({v/n:.1%})")
    if len(na) > limit:
        print(f"    ... ({len(na) - limit} more)")


def build_firm_master_bi(df: pd.DataFrame) -> pd.DataFrame:
    # Strict schema enforcement (no labels applied to file)
    out = apply_gold_schema(df, "firm_master_bi", SCHEMAS_PATH)
    return out


def build_firm_timeseries_bi(df: pd.DataFrame) -> pd.DataFrame:
    # Strict schema enforcement (no labels applied to file)
    out = apply_gold_schema(df, "firm_timeseries_bi", SCHEMAS_PATH)
    return out


def main():
    print("=== Building GOLD (strict) + BI previews (human-friendly) ===")
    print(f"Schemas:    {SCHEMAS_PATH}")
    print(f"Labels YML: {LABEL_YAML} {'(FOUND)' if LABEL_YAML.exists() else '(MISSING – previews & prints use raw names)'}")
    print(f"SILVER_DIR: {SILVER_DIR}")
    print(f"GOLD_DIR:   {GOLD_DIR}")
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    if not SCHEMAS_PATH.exists():
        print(f"[FATAL] schemas.yaml not found at: {SCHEMAS_PATH}"); sys.exit(1)

    successes, failures = [], []

    # 1) Load SILVER and prep keys
    try:
        print("\n[INFO] Loading SILVER (partition scan)...")
        df_silver = load_silver(str(SILVER_DIR))
        print(f"  SILVER shape: {df_silver.shape}")
        _na_report(df_silver, "SILVER (raw)", limit=15)

        df_silver = prepare_keys(df_silver)
        print("  SILVER keys prepared (CRD/REPORT_DATE/DATESUBMITTED/FILING_ID)")
    except Exception as e:
        print(f"[FATAL] Failed to load/prepare SILVER: {e}")
        sys.exit(1)

    # 2) Build logical sources (raw names)
    try:
        print("\n[INFO] Building firm_master (latest per CRD)...")
        firm_master = build_firm_master(df_silver)
        print(f"  firm_master (pre-schema) shape: {firm_master.shape}")
        _na_report(firm_master, "firm_master (pre-schema)", limit=15)
    except Exception as e:
        failures.append(("firm_master_build", str(e))); firm_master = None
        print(f"[ERROR] firm_master build failed: {e}")

    try:
        print("\n[INFO] Building firm_timeseries (latest per period)...")
        firm_timeseries = build_timeseries(df_silver)
        print(f"  firm_timeseries (pre-schema) shape: {firm_timeseries.shape}")
        _na_report(firm_timeseries, "firm_timeseries (pre-schema)", limit=15)
    except Exception as e:
        failures.append(("firm_timeseries_build", str(e))); firm_timeseries = None
        print(f"[ERROR] firm_timeseries build failed: {e}")

    # 3) Strict GOLD (schema-enforced, canonical snake_case on disk)
    try:
        if firm_master is not None:
            print("\n[INFO] Enforcing schema → firm_master_bi (strict)")
            firm_master_bi = build_firm_master_bi(firm_master)
            outp = GOLD_DIR / "firm_master_bi.parquet"
            write_parquet(outp, firm_master_bi)
            print(f"[OK] firm_master_bi → {outp} | shape: {firm_master_bi.shape} | size: {_size_mb(outp)}")
            _na_report(firm_master_bi, "firm_master_bi (strict)", limit=15)
            successes.append("firm_master_bi")
    except Exception as e:
        failures.append(("firm_master_bi", str(e)))
        print(f"[ERROR] firm_master_bi failed: {e}")

    try:
        if firm_timeseries is not None:
            print("\n[INFO] Enforcing schema → firm_timeseries_bi (strict)")
            firm_timeseries_bi = build_firm_timeseries_bi(firm_timeseries)
            outp = GOLD_DIR / "firm_timeseries_bi.parquet"
            write_parquet(outp, firm_timeseries_bi)
            print(f"[OK] firm_timeseries_bi → {outp} | shape: {firm_timeseries_bi.shape} | size: {_size_mb(outp)}")
            _na_report(firm_timeseries_bi, "firm_timeseries_bi (strict)", limit=15)
            successes.append("firm_timeseries_bi")
    except Exception as e:
        failures.append(("firm_timeseries_bi", str(e)))
        print(f"[ERROR] firm_timeseries_bi failed: {e}")

    # 4) BI-labeled preview files (for exploration only; no schema enforcement)
    try:
        if firm_master is not None:
            print("\n[INFO] Creating BI-labeled preview → firm_master_preview.parquet")
            fm_prev = _label_df_for_display(firm_master)
            p = PREVIEW_DIR / "firm_master_preview.parquet"
            write_parquet(p, fm_prev)
            print(f"[OK] firm_master_preview → {p} | shape: {fm_prev.shape} | size: {_size_mb(p)}")
            _na_report(fm_prev, "firm_master_preview", limit=15)
    except Exception as e:
        failures.append(("firm_master_preview", str(e)))
        print(f"[ERROR] firm_master_preview failed: {e}")

    try:
        if firm_timeseries is not None:
            print("\n[INFO] Creating BI-labeled preview → firm_timeseries_preview.parquet")
            ts_prev = _label_df_for_display(firm_timeseries)
            p = PREVIEW_DIR / "firm_timeseries_preview.parquet"
            write_parquet(p, ts_prev)
            print(f"[OK] firm_timeseries_preview → {p} | shape: {ts_prev.shape} | size: {_size_mb(p)}")
            _na_report(ts_prev, "firm_timeseries_preview", limit=15)
    except Exception as e:
        failures.append(("firm_timeseries_preview", str(e)))
        print(f"[ERROR] firm_timeseries_preview failed: {e}")

    # 5) Summary
    print("\n=== Summary ===")
    print(f"Successes: {successes}")
    if failures:
        print("Failures:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print("All GOLD tables + previews built successfully.")


if __name__ == "__main__":
    main()
