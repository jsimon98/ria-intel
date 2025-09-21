# etl_utils.py
from __future__ import annotations
import os, re, glob, unicodedata
from pathlib import Path
from typing import Dict, List
import pandas as pd

# ----------------------------
# Minimal column normalization
# ----------------------------
_CANON_MAP = {
    "FILINGID":"FILING_ID","FILING_ID":"FILING_ID","FILINGNUMBER":"FILING_ID","FILING_NO":"FILING_ID","FILING-NUMBER":"FILING_ID",
    "FIRMCRD":"CRD","FIRM_CRD":"CRD","CRD":"CRD",
    "SEC#":"SEC","SEC_NUMBER":"SEC","SEC":"SEC",
    "FIRMNAME":"FIRM_NAME","FIRM_NAME":"FIRM_NAME",
}

def _canon(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s).replace("\ufeff",""))
    s = re.sub(r"[^0-9A-Za-z]+","_", s.strip()).strip("_")
    return s.upper()

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [_CANON_MAP.get(_canon(c), _canon(c)) for c in df.columns]
    seen: Dict[str,int] = {}; out = []
    for c in cols:
        if c in seen:
            seen[c] += 1; out.append(f"{c}__{seen[c]}")
        else:
            seen[c] = 0; out.append(c)
    df = df.copy()
    df.columns = out
    return df

def read_csv_norm(path: str, encoding: str = "latin1") -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding, low_memory=False, dtype=str)
    return normalize_cols(df)

# ----------------------------
# Basic validations & merging
# ----------------------------
def require_keys(df: pd.DataFrame, keys: List[str]):
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

def enforce_nonblank(df: pd.DataFrame, cols: List[str], label: str = ""):
    for c in cols:
        if df[c].isna().any() or df[c].astype(str).str.strip().eq("").any():
            raise RuntimeError(f"Blank values in required column {c}{' '+label if label else ''}")

def perfect_merge(left: pd.DataFrame, right: pd.DataFrame, on: str | List[str], how: str = "inner", suffixes=("_A","_B")) -> pd.DataFrame:
    require_keys(left, [on] if isinstance(on,str) else on)
    require_keys(right,[on] if isinstance(on,str) else on)
    enforce_nonblank(left, [on] if isinstance(on,str) else on, "(left)")
    enforce_nonblank(right,[on] if isinstance(on,str) else on, "(right)")
    m = left.merge(right, on=on, how=how, suffixes=suffixes)
    if len(m)!=len(left) or len(m)!=len(right):
        raise RuntimeError(f"Imperfect merge on {on}: L={len(left)} R={len(right)} M={len(m)}")
    return m

# ----------------------------
# Helpers
# ----------------------------
def parse_period_from_path(path: str) -> str:
    m = re.search(r"_(\d{8})_(\d{8})\.csv$", path)
    if not m:
        raise RuntimeError(f"Cannot parse period from: {path}")
    return m.group(2)

# ----------------------------
# Simple loader for SILVER
# ----------------------------
# Env var: RIA_SILVER_DIR (no defaults baked into code)
def load_silver(silver_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load partitioned SILVER data for ad-hoc analysis in notebooks.

    Looks for parquet under {silver_dir}/**/*.parquet; if none found, falls
    back to CSVs named 'part.csv' under partition folders.

    Adds a '__source_file' column indicating the file basename.
    """
    base = Path(silver_dir or os.environ.get("RIA_SILVER_DIR","")).expanduser().resolve()
    if not base.exists():
        raise RuntimeError("SILVER directory not found. Set RIA_SILVER_DIR or pass silver_dir.")

    files = glob.glob(str(base / "**" / "*.parquet"), recursive=True)
    use_parquet = bool(files)

    if not files:
        files = glob.glob(str(base / "**" / "part.csv"), recursive=True)
        if not files:
            raise RuntimeError(f"No parquet or CSV parts found under: {base}")

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f) if use_parquet else pd.read_csv(f, dtype=str, low_memory=False)
        except Exception as e:
            raise RuntimeError(f"Failed reading {f}: {e}") from e
        x = df.copy()
        x["__source_file"] = os.path.basename(f)
        dfs.append(x)

    return pd.concat(dfs, ignore_index=True)
