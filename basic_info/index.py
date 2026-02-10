import matplotlib.pyplot as plt

BINS = 50

def basic_info(df):
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


def save_hist(series, title, xlabel, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(series.dropna(), bins=BINS)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
