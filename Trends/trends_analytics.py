from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List
import json
import time

import pandas as pd
from Trends.trends_plot import plot_trends


@dataclass(frozen=True)
class Config:
    source_system: str
    chunksize: int = 200_000

    keep_csv_cols: tuple[str, ...] = ("Timestamp", "SourceSystem", "LatencyMs", "AttributesJson")
    json_keys: tuple[str, ...] = ("TrendWindow", "CpuUsage", "MemoryUsageMb", "DiskQueueLength")

    # etap 5
    metric: str = "LatencyMs"        # "LatencyMs" lub np. "CpuUsage" / "MemoryUsageMb" / "DiskQueueLength"
    bucket_seconds: int = 60         # np. 30, 60, 300
    agg: str = "median"              # "median" albo "mean"

    # etap 6 - detekcja trendu
    smooth_window: int = 5      # liczba bucketów do wygładzenia
    min_run: int = 5            # min. liczba kolejnych wzrostów
    min_delta: float = 0.0      # minimalny przyrost (0 = dowolny wzrost)


def _safe_json_loads(s: Any) -> Optional[Dict[str, Any]]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

def _parse_timestamp_utc(ts: pd.Series) -> pd.Series:
    s = ts.astype("string").str.strip()
    try:
        return pd.to_datetime(s, format="ISO8601", utc=True, errors="coerce")
    except Exception:
        s = s.str.replace("Z", "+00:00", regex=False)
        return pd.to_datetime(s, utc=True, errors="coerce")

def prepare_dataframe(df_in: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Filter by SourceSystem
    if "SourceSystem" in df_in.columns:
        df = df_in[df_in["SourceSystem"] == cfg.source_system].copy()
    else:
        # If no SourceSystem column, assume it's already filtered or proceed with caution
        df = df_in.copy()

    if df.empty:
        return df

    # Timestamp parsing
    if "Timestamp" in df.columns:
         # Check if already datetime
        if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
             df["Timestamp"] = _parse_timestamp_utc(df["Timestamp"])
        
        df = df.dropna(subset=["Timestamp"])
    
    if df.empty:
        return df

    # Ensure types for metrics and TrendWindow
    if "TrendWindow" in df.columns:
        df["TrendWindow"] = df["TrendWindow"].astype("boolean")
    
    for col in ("CpuUsage", "MemoryUsageMb", "DiskQueueLength", "LatencyMs"):
        if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def bucketize(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["bucket", "metric_agg", "TrendWindow_bucket"])

    if cfg.metric not in df.columns:
        # If metric column is missing, try to see if we can use it, otherwise return empty
        # But maybe prints will handle it
        print(f"Brak kolumny metryki: {cfg.metric}")
        return pd.DataFrame(columns=["bucket", "metric_agg", "TrendWindow_bucket"])

    work = df[["Timestamp", cfg.metric]].copy()
    
    if "TrendWindow" in df.columns:
        work["TrendWindow"] = df["TrendWindow"]
    else:
        work["TrendWindow"] = False

    work[cfg.metric] = pd.to_numeric(work[cfg.metric], errors="coerce")
    work["TrendWindow"] = work["TrendWindow"].fillna(False)

    # indeks czasowy do resample
    work = work.set_index("Timestamp").sort_index()

    rule = f"{cfg.bucket_seconds}s"

    if cfg.agg == "median":
        metric_agg = work[cfg.metric].resample(rule).median()
    elif cfg.agg == "mean":
        metric_agg = work[cfg.metric].resample(rule).mean()
    else:
        raise ValueError("cfg.agg musi być 'median' albo 'mean'")

    # ground truth na poziomie bucketu: jeśli w bucket był choć 1 event z TrendWindow=True
    tw_bucket = work["TrendWindow"].resample(rule).max().astype("boolean")

    out = pd.DataFrame(
        {
            "bucket": metric_agg.index,
            "metric_agg": metric_agg.values,
            "TrendWindow_bucket": tw_bucket.values,
        }
    )

    # usuń buckety bez metryki (np. wszystkie NaN w oknie)
    out = out.dropna(subset=["metric_agg"]).reset_index(drop=True)
    return out


def detect_trend(cfg: Config, buckets: pd.DataFrame) -> pd.DataFrame:
    if buckets.empty:
        return buckets.assign(
            metric_smooth=pd.Series(dtype="float64"),
            is_trend=pd.Series(dtype="boolean"),
        )

    work = buckets.copy()

    # wygładzenie
    if cfg.agg == "median":
        work["metric_smooth"] = work["metric_agg"].rolling(
            window=cfg.smooth_window, min_periods=cfg.smooth_window
        ).median()
    else:
        work["metric_smooth"] = work["metric_agg"].rolling(
            window=cfg.smooth_window, min_periods=cfg.smooth_window
        ).mean()

    # różnice
    d = work["metric_smooth"].diff()

    # punkt wzrostowy
    inc = d > cfg.min_delta

    # wykrycie ciągów wzrostów długości >= min_run
    run_id = (inc != inc.shift(1, fill_value=False)).cumsum()
    run_len = inc.groupby(run_id).transform("sum")

    work["is_trend"] = inc & (run_len >= cfg.min_run)
    work["is_trend"] = work["is_trend"].astype("boolean")

    return work

def run_analysis(df_in: pd.DataFrame, output_dir: str = ".") -> None:
    print("\n--- Analiza Trendów (Trends Analytics) ---")
    cfg = Config(
        source_system="OrderService",
        chunksize=200_000,
        metric="LatencyMs",      # możesz zmienić na "CpuUsage"/"MemoryUsageMb"/"DiskQueueLength"
        bucket_seconds=120,
        agg="median",          
        smooth_window=7,
        min_run=7,
        min_delta=0.0,  
    )

    t0 = time.perf_counter()
    
    # Prepare the dataframe (filter, type cast)
    df = prepare_dataframe(df_in, cfg)
    
    if df.empty:
        print(f"[Trends] Brak danych dla SourceSystem={cfg.source_system} lub pusty DataFrame.")
        return

    buckets = bucketize(cfg, df)
    det = detect_trend(cfg, buckets)

    # Check if TrendWindow exists
    if "TrendWindow" in df.columns:
        trend_true = int(df["TrendWindow"].fillna(False).sum())
        trend_segments = int(((df["TrendWindow"].fillna(False)) & (~df["TrendWindow"].fillna(False).shift(1, fill_value=False))).sum())
    else:
        trend_true = -1
        trend_segments = -1

    # ground truth i predykcja
    if "TrendWindow_bucket" in det.columns and "is_trend" in det.columns:
        y_true = det["TrendWindow_bucket"].fillna(False)
        y_pred = det["is_trend"].fillna(False)

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        y_pred = pd.Series([False]*len(det)) if not det.empty else pd.Series([], dtype="bool")

    elapsed = time.perf_counter() - t0

    print(f"System źródłowy: {cfg.source_system}")
    print(f"Liczba wierszy (po filtracji): {len(df)}")
    if trend_true >= 0:
        print(f"TrendWindow == True: {trend_true}")
        print(f"Liczba przedziałów TrendWindow == True (events): {trend_segments}")
    
    print(f"Buckets: {len(buckets)}  bucket_seconds={cfg.bucket_seconds}  metric={cfg.metric}  agg={cfg.agg}")
    print(f"Wykryte buckety trendu: {int(y_pred.sum())}")
    
    if "TrendWindow_bucket" in det.columns:
        print(f"TrendWindow == True (buckets): {int(det['TrendWindow_bucket'].sum())}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1:        {f1:.3f}")    
    
    print(f"Czas wykonania [s]: {elapsed:.2f}")
    
    # Generate visualization
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_trends(det, cfg, output_path=str(out_dir / "trend_analysis.png"))

if __name__ == "__main__":
    # For testing, assumes main.py logic creates df
    print("Run from main.py please.")
