import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BINS = 50
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = ROOT_DIR / "SyntheticLogGenerator" / "logs.csv"
DEFAULT_LIMIT = 200000


def basic_info(df, bins=BINS, plots_dir="."):
    df = df.copy()

    if "AttributesJson" in df.columns:
        attrs = df["AttributesJson"]
        if len(attrs) > 0 and isinstance(attrs.iloc[0], str):
            attrs = attrs.fillna("{}").apply(json.loads)
        attrs_df = pd.json_normalize(attrs)
        df = pd.concat([df.drop(columns=["AttributesJson"]), attrs_df], axis=1)

    print("\n=== Podstawowe statystyki ===")
    print("Liczba eventów:", len(df))
    print("Liczba transakcji (TransactionId):", df["TransactionId"].nunique())
    print("Liczba korelacji (CorrelationId):", df["CorrelationId"].nunique())

    print("\n=== Rozkład Priority ===")
    print(df["Priority"].value_counts())

    print("\n=== Rozkład EventCode ===")
    print(df["EventCode"].value_counts().sort_index())

    print("\n=== Rozkład SourceSystem ===")
    print(df["SourceSystem"].value_counts())

    print("\n=== Rozkład Scenario ===")
    print(df["Scenario"].value_counts())

    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    save_hist(df["LatencyMs"], "LatencyMs", "Latency [ms]", f"{plots_dir}/latency_hist.png", bins)
    save_hist(df["CpuUsage"], "CpuUsage", "CPU usage [%]", f"{plots_dir}/cpu_usage_hist.png", bins)
    save_hist(df["LocalQps"], "LocalQps", "Local QPS", f"{plots_dir}/local_qps_hist.png", bins)
    save_hist(
        df["DiskQueueLength"],
        "DiskQueueLength",
        "Disk queue length",
        f"{plots_dir}/disk_queue_hist.png",
        bins,
    )

    return {
        "events_count": int(len(df)),
        "transactions_count": int(df["TransactionId"].nunique()),
        "correlations_count": int(df["CorrelationId"].nunique()),
    }


def save_hist(series, title, xlabel, filename, bins=BINS):
    values = pd.to_numeric(series, errors="coerce").dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    test_df = pd.read_csv(DEFAULT_CSV_PATH, sep=";", nrows=DEFAULT_LIMIT)
    basic_info(test_df, bins=BINS, plots_dir=".")
