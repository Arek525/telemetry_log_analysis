from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
import os

# Suppress joblib/loky warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

FEATURE_COLS = [
    "LatencyMs",
    "CpuUsage",
    "MemoryUsageMb",
    "DiskQueueLength",
    "NetworkErrors",
    "LocalQps",
    "RequestSizeBytes",
    "ResponseSizeBytes",
]

def build_feature_matrix(df: pd.DataFrame):
    # Check which feature cols exist
    existing_cols = [c for c in FEATURE_COLS if c in df.columns]
    
    if not existing_cols:
        return pd.DataFrame(), np.array([])

    df_features = df[existing_cols].copy()
    
    # Ensure numeric
    for col in existing_cols:
        df_features[col] = pd.to_numeric(df_features[col], errors="coerce")

    df_features = df_features.dropna()

    X = df_features.values
    return df_features, X


def standardize_features(X: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def run_kmeans(X_scaled, k_values=(3, 5, 8)):
    results = {}

    for k in k_values:
        t0 = time.perf_counter()

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(X_scaled)

        elapsed = time.perf_counter() - t0

        cluster_sizes = pd.Series(labels).value_counts().sort_index()

        print("\n==============================")
        print(f"K = {k}")
        print(f"Rozmiary klastrów:\n{cluster_sizes}")
        # print(f"Silhouette score: {silhouette:.4f}")
        print(f"KMeans czas [s]: {elapsed:.2f}")

        results[k] = {
            "model": model,
            "labels": labels,
        }

    return results

def analyze_clusters_tables(df, df_features, results, k_values=(3, 5, 8)):
    base = df.loc[df_features.index].copy()

    for k in k_values:
        print("\n========================================")
        print(f"ANALIZA KLASTRÓW DLA K = {k}")
        print("========================================")

        labels = results[k]["labels"]
        df_k = base.copy()
        df_k["cluster"] = labels

        # Ensure types for analysis columns if they exist
        if "EventCode" in df_k.columns:
            df_k["is_failure"] = df_k["EventCode"].isin([500, 501])
            df_k["is_anomaly_code"] = df_k["EventCode"].isin([998, 999])
        else:
            df_k["is_failure"] = False
            df_k["is_anomaly_code"] = False

        # ===============================
        # TABELA 1 - ŚREDNIE
        # ===============================

        counts = df_k.groupby("cluster").size().to_frame("count")

        current_feature_cols = [c for c in FEATURE_COLS if c in df_k.columns]

        means = (
            df_k
            .groupby("cluster")[current_feature_cols]
            .mean()
            .round(2)
        )

        table_means = counts.join(means).sort_index()

        print("\n--- TABELA 1: ŚREDNIE CECH ---")
        print(table_means.to_string())

        # ===============================
        # TABELA 2 - PROCENTY
        # ===============================

        pct_cols = []
        if "IsAnomaly" in df_k.columns: pct_cols.append("IsAnomaly")
        pct_cols.append("is_failure")
        pct_cols.append("is_anomaly_code")

        for c in ["IsSpikeByLatency", "IsSpikeByCpu", "IsSpikeByQps"]:
            if c in df_k.columns:
                pct_cols.append(c)

        # Check if pct_cols exist in df_k
        valid_pct_cols = [c for c in pct_cols if c in df_k.columns]

        if valid_pct_cols:
            pct = (
                df_k
                .groupby("cluster")[valid_pct_cols]
                .mean()
                .round(4)
            )

            pct = pct.rename(
                columns={
                    "IsAnomaly": "pct_IsAnomaly",
                    "is_failure": "pct_EventCode_500_501",
                    "is_anomaly_code": "pct_EventCode_998_999",
                    "IsSpikeByLatency": "pct_IsSpikeByLatency",
                    "IsSpikeByCpu": "pct_IsSpikeByCpu",
                    "IsSpikeByQps": "pct_IsSpikeByQps",
                }
            )

            table_pct = counts.join(pct).sort_index()

            print("\n--- TABELA 2: PROCENTY ---")
            print(table_pct.to_string())

def run_analysis(df: pd.DataFrame):
    print("\n--- Analiza K-Means (K-Means Analytics) ---")
    t0 = time.perf_counter()

    # ETAP 2
    df_features, X = build_feature_matrix(df)
    
    if X.size == 0:
        print("[KMeans] Brak danych do klasteryzacji (pusta macierz cech).")
        return

    # ETAP 3
    X_scaled, scaler = standardize_features(X)
    print(f"\nCzas przygotowania danych [s]: {(time.perf_counter()-t0):.2f}")

    # ETAP 4
    t1 = time.perf_counter()
    results = run_kmeans(X_scaled, k_values=(3, 5, 8))
    print(f"\nCzas klasteryzacji (wszystkie K) [s]: {(time.perf_counter()-t1):.2f}")

    # ETAP 5
    t1 = time.perf_counter()
    analyze_clusters_tables(df, df_features, results, k_values=(3, 5, 8))
    print(f"\nCzas analizy tabel [s]: {(time.perf_counter()-t1):.2f}")

    elapsed = time.perf_counter() - t0

    print(f"\nLiczba wierszy w analizie: {len(df_features)}")
    print(f"Kształt macierzy cech: {X_scaled.shape}")
    print(f"\nCałkowity czas wykonania [s]: {elapsed:.2f}")

if __name__ == "__main__":
    print("Run from main.py please.")
