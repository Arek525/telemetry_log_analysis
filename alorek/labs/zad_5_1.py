import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# Konfiguracja
# ==============================
CSV_PATH = "../logs.csv"
PLOTS_DIR = "plots"
BINS = 50


def main():
    print("=== Zadanie 5.1 – wstępna eksploracja danych ===")

    # ----------------------------------
    # 1. Wczytanie danych
    # ----------------------------------
    df = pd.read_csv(CSV_PATH, sep=";")
    print(f"Liczba eventów: {len(df)}")

    # ----------------------------------
    # 2. Parsowanie AttributesJson
    # ----------------------------------
    attributes_df = df["AttributesJson"].apply(json.loads).apply(pd.Series)
    df = pd.concat([df.drop(columns=["AttributesJson"]), attributes_df], axis=1)

    # ----------------------------------
    # 3. Podstawowe liczniki
    # ----------------------------------
    print("\n=== Podstawowe statystyki ===")
    print("Liczba eventów:", len(df))
    print("Liczba transakcji (TransactionId):", df["TransactionId"].nunique())
    print("Liczba korelacji (CorrelationId):", df["CorrelationId"].nunique())

    # ----------------------------------
    # 4. Rozkłady kategorialne
    # ----------------------------------
    print("\n=== Rozkład Priority ===")
    print(df["Priority"].value_counts())

    print("\n=== Rozkład EventCode ===")
    print(df["EventCode"].value_counts().sort_index())

    print("\n=== Rozkład SourceSystem ===")
    print(df["SourceSystem"].value_counts())

    print("\n=== Rozkład Scenario ===")
    print(df["Scenario"].value_counts())

    # ----------------------------------
    # 5. Histogramy (WYKRESY)
    # ----------------------------------
    Path(PLOTS_DIR).mkdir(exist_ok=True)

    save_hist(
        df["LatencyMs"],
        title="LatencyMs",
        xlabel="Latency [ms]",
        filename="latency_hist.png"
    )

    save_hist(
        df["CpuUsage"],
        title="CpuUsage",
        xlabel="CPU usage [%]",
        filename="cpu_usage_hist.png"
    )

    save_hist(
        df["LocalQps"],
        title="LocalQps",
        xlabel="Local QPS",
        filename="local_qps_hist.png"
    )

    save_hist(
        df["DiskQueueLength"],
        title="DiskQueueLength",
        xlabel="Disk queue length",
        filename="disk_queue_hist.png"
    )

    print("\nHistogramy zapisane w katalogu:", PLOTS_DIR)
    print("Zadanie 5.1 ZAKOŃCZONE")


def save_hist(series, title, xlabel, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(series.dropna(), bins=BINS)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(Path(PLOTS_DIR) / filename)
    plt.close()


if __name__ == "__main__":
    main()
