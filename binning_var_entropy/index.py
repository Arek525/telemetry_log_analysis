import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = ROOT_DIR / "SyntheticLogGenerator" / "logs.csv"
DEFAULT_LIMIT = None


def b_v_e(df, metric="LatencyMs", bins_count=20):
    # 1) Rozbicie AttributesJson na kolumny
    attrs = df["AttributesJson"].fillna("{}").apply(json.loads)
    attrs_df = pd.json_normalize(attrs)
    df = pd.concat([df.drop(columns=["AttributesJson"]), attrs_df], axis=1)

    # 2) Przygotowanie metryki i binow
    metric_values = pd.to_numeric(df[metric], errors="coerce")
    all_values = metric_values.dropna()
    bins = np.histogram_bin_edges(all_values, bins=bins_count)

    # 3) Okresy z gotowych flag w danych
    spike = (df["SpikeWindow"] == True).fillna(False)
    failure = (df["FailureWindow"] == True).fillna(False)
    trend = (df["TrendWindow"] == True).fillna(False)

    periods = {
        "bez_spike": ~spike,
        "ze_spike": spike,
        "z_failure": failure,
        "z_trendem": trend,
    }

    result = {
        "metric": metric,
        "bins_count": int(len(bins) - 1),
        "bin_edges": bins.tolist(),
        "periods": {},
        "comparison_vs_bez_spike": {},
    }

    print("\n=== Binning, wariancja, entropia, balance ratio ===")
    print(f"Metryka: {metric}, liczba binow: {len(bins) - 1}")
    print("Okresy wziete z flag: SpikeWindow, FailureWindow, TrendWindow")

    # 4) Statystyki dla 4 grup
    for name, mask in periods.items():
        values = pd.to_numeric(df.loc[mask, metric], errors="coerce").dropna()
        counts, _ = np.histogram(values, bins=bins)
        stats = compute_stats(counts)

        result["periods"][name] = {
            "events": int(len(values)),
            "counts": counts.tolist(),
            "variance": stats["variance"],
            "entropy": stats["entropy"],
            "balance_max_min": stats["balance_max_min"],
            "balance_max_median": stats["balance_max_median"],
        }

        print(f"\n--- {name} ---")
        for left, right, count in zip(bins[:-1], bins[1:], counts):
            print(f"[{left:.2f}, {right:.2f}): {count}")
        print(
            f"events={len(values)} | "
            f"wariancja={stats['variance']:.4f} | "
            f"entropia={stats['entropy']:.4f} | "
            f"balance max/min={ratio_text(stats['balance_max_min'])} | "
            f"balance max/mediana={ratio_text(stats['balance_max_median'])}"
        )

    # 5) Porownanie do bez_spike
    base = result["periods"]["bez_spike"]
    print("\nPorownanie wzgledem bez_spike:")
    for name in ["ze_spike", "z_failure", "z_trendem"]:
        p = result["periods"][name]
        variance_ratio = p["variance"] / base["variance"] if base["variance"] > 0 else float("inf")
        entropy_delta = p["entropy"] - base["entropy"]

        result["comparison_vs_bez_spike"][name] = {
            "variance_ratio": variance_ratio,
            "entropy_delta": entropy_delta,
        }
        print(
            f"- {name}: wariancja x{ratio_text(variance_ratio)}, "
            f"delta entropii {entropy_delta:+.4f}"
        )

    return result


def compute_stats(counts):
    total = counts.sum()
    variance = float(np.var(counts)) if len(counts) > 0 else 0.0

    if total > 0:
        p = counts / total
        p = p[p > 0]
        entropy = float(-(p * np.log2(p)).sum())
    else:
        entropy = 0.0

    max_count = int(counts.max()) if len(counts) > 0 else 0
    nonzero = counts[counts > 0]
    min_nonzero = int(nonzero.min()) if len(nonzero) > 0 else 0
    median_count = float(np.median(counts)) if len(counts) > 0 else 0.0

    balance_max_min = float(max_count / min_nonzero) if min_nonzero > 0 else float("inf")
    balance_max_median = float(max_count / median_count) if median_count > 0 else float("inf")

    return {
        "variance": variance,
        "entropy": entropy,
        "balance_max_min": balance_max_min,
        "balance_max_median": balance_max_median,
    }


def ratio_text(x):
    if np.isinf(x):
        return "inf"
    return f"{x:.4f}"


if __name__ == "__main__":
    test_df = pd.read_csv(DEFAULT_CSV_PATH, sep=";", nrows=DEFAULT_LIMIT)
    b_v_e(test_df)
