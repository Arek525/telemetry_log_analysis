import numpy as np

METRIC = "LatencyMs"
BINS = 2


def b_v_e():
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
