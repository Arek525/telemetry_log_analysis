import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

def anomaly_detection(df, out_path="anomaly_detection_results.csv"):
    features = [
        "LatencyMs",
        "CpuUsage",
        "MemoryUsageMb",
        "DiskQueueLength",
        "NetworkErrors",
        "LocalQps",
        "RequestSizeBytes",
        "ResponseSizeBytes",
    ]

    X = df[features].copy().fillna(df[features].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    error_5xx = ((df["EventCode"] >= 500) & (df["EventCode"] < 600)).sum()
    base_rate = error_5xx / len(df)
    contamination = max(0.00001, min(0.5, base_rate / 10))
    print(f"contamination={contamination:.6f} (5xx_rate/10, 5xx={error_5xx})")

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        max_samples="auto",
    )
    preds = model.fit_predict(X_scaled)
    scores = model.score_samples(X_scaled)
    df["IF_Anomaly"] = (preds == -1).astype(int)
    df["IF_AnomalyScore"] = scores

    detected = df["IF_Anomaly"].sum()
    print(f"Wykryto anomalii: {detected}")

    if "IsAnomaly" in df.columns:
        y_true = df["IsAnomaly"].values
        y_pred = df["IF_Anomaly"].values
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    # Top 3 scenariusze wg Isolation Forest
    if "Scenario" in df.columns:
        anomalies = df[df["IF_Anomaly"] == 1]
        print("\n=== TOP 3 SCENARIUSZE Z ANOMALIAMI (IF) ===")
        top_counts = anomalies["Scenario"].value_counts().head(3)
        for i, (name, cnt) in enumerate(top_counts.items(), start=1):
            print(f"{i}. {name}: {cnt}")
        feature_labels = {
            "LatencyMs": "latencja",
            "CpuUsage": "CPU",
            "MemoryUsageMb": "RAM",
            "DiskQueueLength": "dysk",
            "NetworkErrors": "sieć",
        }
        feat_cols = list(feature_labels.keys())
        med = df[feat_cols].median()
        iqr = (df[feat_cols].quantile(0.75) - df[feat_cols].quantile(0.25)).replace(0, np.nan)

        def reason_flags(row):
            z = ((row[feat_cols] - med) / iqr).abs().to_numpy(dtype=float)
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            top_idx = np.argsort(-z)[:2]
            flags = [feature_labels[feat_cols[i]] for i in top_idx if z[i] > 0]
            return flags if flags else ["inne"]

        anomalies = anomalies.copy()
        anomalies["_reasons"] = anomalies.apply(reason_flags, axis=1)
        top_scenarios = anomalies["Scenario"].value_counts().head(3).index
        for scen in top_scenarios:
            subset = anomalies[anomalies["Scenario"] == scen]
            reasons = subset["_reasons"].explode().value_counts().head(5)
            print(f"\n--- Scenario: {scen} ---")
            print("Powody anomalii (top):")
            for label, cnt in reasons.items():
                print(f"- {label}: {cnt}")

            dominant = reasons.index[0] if len(reasons) > 0 else "inne"
            print("Rekomendacje:")
            if dominant == "latencja":
                print("- sprawdź p95/p99 opóźnienia per endpoint i zależności")
                print("- zweryfikuj timeouts i kolejki requestów")
            elif dominant == "sieć":
                print("- sprawdź timeouty, retry storm i błędy połączeń")
                print("- zweryfikuj obciążenie downstream i limity połączeń")
            elif dominant == "dysk":
                print("- sprawdź IO wait, opóźnienia storage/DB i kolejki")
                print("- ogranicz synchroniczne zapisy i logowanie")
            elif dominant == "CPU":
                print("- sprawdź hot-pathy, GC i wykorzystanie CPU per endpoint")
                print("- rozważ skalowanie poziome lub optymalizację CPU")
            elif dominant == "RAM":
                print("- sprawdź wycieki pamięci i presję GC")
                print("- zweryfikuj limity i alokacje obiektów")
            else:
                print("- przeanalizuj kombinacje cech i korelacje z ruchem")
    return df
