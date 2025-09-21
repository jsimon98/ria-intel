# scripts/merge_ia_base_ab.py
import re, glob, os, pandas as pd
from pathlib import Path
from riaintel.etl.etl_utils import read_csv_norm, require_keys, enforce_nonblank, perfect_merge

RAW     = os.getenv("RIA_RAW_DIR")
RUNTIME = os.getenv("RIA_RUNTIME_DIR")
SILVER  = os.getenv("RIA_SILVER_DIR") or os.path.join(RUNTIME, "silver")
YAML    = os.getenv("RIA_YAML_PATH")

if not RAW or not RUNTIME or not YAML:
    raise RuntimeError("Environment variables RIA_RAW_DIR, RIA_RUNTIME_DIR, RIA_YAML_PATH must be set")

period = lambda p: re.search(r"_(\d{8})_(\d{8})\.csv$", p).group(2)
def prefer_original(paths): 
    return sorted(paths, key=lambda x: ("(1)" in os.path.basename(x), len(os.path.basename(x))))[0]

a_files = glob.glob(f"{RAW}\\IA_ADV_Base_A_*.csv")
b_files = glob.glob(f"{RAW}\\IA_ADV_Base_B_*.csv")
a_by_period = {}
b_by_period = {}
for p in a_files:
    m = re.search(r"_(\d{8})_(\d{8})\.csv$", p)
    if m: a_by_period.setdefault(m.group(2), []).append(p)
for p in b_files:
    m = re.search(r"_(\d{8})_(\d{8})\.csv$", p)
    if m: b_by_period.setdefault(m.group(2), []).append(p)

periods = sorted(set(a_by_period.keys()) | set(b_by_period.keys()))

frames=[]
for perid in periods:
    if perid not in a_by_period or perid not in b_by_period:
        print(f"⚠️ Skipping {perid} — missing A or B file")
        continue
    A = read_csv_norm(prefer_original(a_by_period[perid]))
    B = read_csv_norm(prefer_original(b_by_period[perid]))
    require_keys(A, ["FILING_ID"]); require_keys(B, ["FILING_ID"])
    enforce_nonblank(A, ["FILING_ID"], "(Base A)")
    enforce_nonblank(B, ["FILING_ID"], "(Base B)")
    M = perfect_merge(A, B, on="FILING_ID", suffixes=("_A","_B"))
    M["REPORT_DATE"]=perid
    print(f"✅ {perid}: {len(M)} rows")
    frames.append(M)

ia_base_merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
print(f"📊 rows={len(ia_base_merged)} | periods={len(frames)} | shape={ia_base_merged.shape}")

if ia_base_merged.empty: raise RuntimeError("No data to write to silver")

d = pd.to_datetime(ia_base_merged["REPORT_DATE"], format="%Y%m%d", errors="raise")
ia_base_merged["report_year"]=d.dt.year
ia_base_merged["report_month"]=d.dt.month

try:
    import pyarrow as pa, pyarrow.dataset as ds
    Path(SILVER).mkdir(parents=True, exist_ok=True)
    t = pa.Table.from_pandas(ia_base_merged, preserve_index=False)
    ds.write_dataset(t, base_dir=SILVER, format="parquet",
                     partitioning=["report_year","report_month"],
                     existing_data_behavior="overwrite_or_ignore")
    print(f"💾 wrote SILVER parquet → {SILVER}")
except ImportError:
    for (y,m), g in ia_base_merged.groupby(["report_year","report_month"]):
        p = Path(SILVER) / f"report_year={y}" / f"report_month={m}"
        p.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["report_year","report_month"]).to_csv(p/"part.csv", index=False)
    print(f"💾 wrote SILVER CSV fallback → {SILVER}")

yml = Path(YAML)
if yml.exists():
    import yaml
    meta = yaml.safe_load(yml.read_text(encoding="utf-8"))["columns"]
    rename = {k:v["label"] for k,v in meta.items() if k in ia_base_merged.columns}
    bi = ia_base_merged.rename(columns=rename)
    bi_dir = os.path.join(RUNTIME, "silver_bi")
    try:
        import pyarrow as pa, pyarrow.dataset as ds
        Path(bi_dir).mkdir(parents=True, exist_ok=True)
        tb = pa.Table.from_pandas(bi, preserve_index=False)
        ds.write_dataset(tb, base_dir=bi_dir, format="parquet",
                         partitioning=["report_year","report_month"],
                         existing_data_behavior="overwrite_or_ignore")
        print(f"💾 wrote SILVER_BI parquet → {bi_dir}")
    except ImportError:
        for (y,m), g in bi.groupby(["report_year","report_month"]):
            p = Path(bi_dir) / f"report_year={y}" / f"report_month={m}"
            p.mkdir(parents=True, exist_ok=True)
            g.drop(columns=["report_year","report_month"]).to_csv(p/"part.csv", index=False)
        print(f"💾 wrote SILVER_BI CSV fallback → {bi_dir}")
else:
    print(f"ℹ️ No YAML at {YAML} — skipped BI-labeled silver")
