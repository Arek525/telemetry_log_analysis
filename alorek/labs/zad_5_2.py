from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT_DIR / "SyntheticLogGenerator" / "logs.csv"
METRIC = "LatencyMs"
BINS = 2


def main():
    print("=== Zadanie 5.2 - binning ===")
    print(f"Plik: {CSV_PATH}")
    print(f"Metryka: {METRIC}, liczba binow: {BINS}")
    df = pd.read_csv(CSV_PATH, sep=";", usecols=[METRIC])
    print(f"Liczba eventow: {len(df)}")

    metric_all = pd.to_numeric(df[METRIC], errors="coerce").dropna()
    bins = np.histogram_bin_edges(metric_all, bins=BINS)
    counts, _ = np.histogram(metric_all, bins=bins)

    print("Binning (bin -> liczba eventow):")
    print_counts(bins, counts)
    stats = compute_stats(counts)
    print(
        f"Wariancja: {stats['variance']:.4f} | "
        f"Entropia: {stats['entropy']:.4f} | "
        f"Balance max/min: {stats['balance_max_min']} | "
        f"Balance max/mediana: {stats['balance_max_median']}"
    )
    print("\nZadanie 5.2 ZAKONCZONE")


def compute_stats(counts: np.ndarray) -> dict:
    total = counts.sum()
    variance = float(np.var(counts)) if len(counts) else 0.0

    if total > 0:
        p = counts / total
        p = p[p > 0]
        entropy = float(-(p * np.log2(p)).sum())
    else:
        entropy = 0.0

    max_count = int(counts.max()) if len(counts) else 0
    nonzero = counts[counts > 0]
    min_nonzero = int(nonzero.min()) if len(nonzero) else 0
    median_count = float(np.median(counts)) if len(counts) else 0.0

    balance_max_min = (
        f"{max_count / min_nonzero:.4f}" if min_nonzero > 0 else "inf"
    )
    balance_max_median = (
        f"{max_count / median_count:.4f}" if median_count > 0 else "inf"
    )

    return {
        "variance": variance,
        "entropy": entropy,
        "balance_max_min": balance_max_min,
        "balance_max_median": balance_max_median,
    }


def print_counts(bins: np.ndarray, counts: np.ndarray) -> None:
    for left, right, count in zip(bins[:-1], bins[1:], counts):
        print(f"[{left:.2f}, {right:.2f}): {count}")


if __name__ == "__main__":
    main()
