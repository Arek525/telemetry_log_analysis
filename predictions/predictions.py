import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error, r2_score


def predictions(data, seq_id="CorrelationId", n_prefix=3, test_size=0.2, random_state=42):
    start_time = time.time()
    df = data.copy()

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)

    for c in ["EventCode", "LatencyMs", "IsAnomaly"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "IsAnomaly" in df.columns:
        df["IsAnomaly"] = df["IsAnomaly"].fillna(0).astype(int)

    if "AttributesJson" in df.columns:
        attrs = df["AttributesJson"].fillna("{}").map(
            lambda s: json.loads(s) if isinstance(s, str) and s.strip().startswith("{") else {}
        )
        attrs_df = pd.json_normalize(attrs)
        df = pd.concat([df.drop(columns=["AttributesJson"], errors="ignore"), attrs_df], axis=1)

    leak_cols = [c for c in df.columns if c.lower() in {"isspikebylatency", "isspikebycpu", "isspikebyqps", "spikewindow", "failurewindow", "trendwindow"}]
    df = df.drop(columns=leak_cols, errors="ignore")

    if "EventCode" not in df.columns:
        raise ValueError("Brak kolumny EventCode")
    if "LatencyMs" not in df.columns:
        raise ValueError("Brak kolumny LatencyMs")

    df["is_failure"] = df["EventCode"].isin([500, 501]).astype(int)
    df["is_anomaly_target"] = ((df.get("IsAnomaly", 0) == 1) | (df["EventCode"].isin([998, 999]))).astype(int)

    drop_for_features = {"EventCode", "IsAnomaly", "is_failure", "is_anomaly_target", "LatencyMs"}
    X_event = df.drop(columns=[c for c in drop_for_features if c in df.columns], errors="ignore")

    num_cols = [c for c in X_event.columns if pd.api.types.is_numeric_dtype(X_event[c])]
    cat_cols = [c for c in X_event.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    def clf_metrics(y_true, y_pred):
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        return float(p), float(r), float(f1)

    def split_fit_eval_classifier(y):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_event, y, test_size=test_size, random_state=random_state, stratify=y
        )
        model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        return clf_metrics(y_te, pred)

    def split_fit_eval_regressor(y):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_event, y, test_size=test_size, random_state=random_state
        )
        model = Pipeline([("pre", pre), ("reg", Ridge(alpha=1.0))])
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        mae = mean_absolute_error(y_te, pred)
        mse = mean_squared_error(y_te, pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_te, pred)
        return float(mae), float(rmse), float(r2)

    failure_p, failure_r, failure_f1 = split_fit_eval_classifier(df["is_failure"])
    anomaly_p, anomaly_r, anomaly_f1 = split_fit_eval_classifier(df["is_anomaly_target"])
    lat_mae, lat_rmse, lat_r2 = split_fit_eval_regressor(df["LatencyMs"])

    seq_cols_drop = {"is_failure", "is_anomaly_target", "LatencyMs"}
    seq_base = df.drop(columns=[c for c in seq_cols_drop if c in df.columns], errors="ignore").copy()

    if seq_id not in seq_base.columns:
        seq_p = seq_r = seq_f1 = np.nan
    else:
        if "Timestamp" in seq_base.columns:
            seq_base = seq_base.sort_values(["Timestamp"])
        else:
            seq_base = seq_base.reset_index(drop=True)

        seq_base["_step"] = seq_base.groupby(seq_id).cumcount()
        seq_prefix = seq_base[seq_base["_step"] < n_prefix].copy()

        y_tx = df.groupby(seq_id, dropna=False).apply(
            lambda g: int(g["EventCode"].isin([500, 501, 998, 999]).any() or (g.get("IsAnomaly", 0) == 1).any())
        ) if seq_id in df.columns else None

        tx_features = []
        for tx, g in seq_prefix.groupby(seq_id):
            row = {}
            row[seq_id] = tx
            row["events_in_prefix"] = len(g)
            for col in ["CpuUsage", "LocalQps", "Retries", "DiskQueueLength", "NetworkErrors", "MemoryUsageMb"]:
                if col in g.columns:
                    vals = pd.to_numeric(g[col], errors="coerce")
                    row[f"{col}_mean"] = float(np.nanmean(vals)) if np.isfinite(np.nanmean(vals)) else np.nan
                    row[f"{col}_max"] = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else np.nan
            if "SourceSystem" in g.columns:
                row["unique_services"] = int(g["SourceSystem"].nunique(dropna=True))
            if "Scenario" in g.columns:
                row["unique_scenarios"] = int(g["Scenario"].nunique(dropna=True))
            tx_features.append(row)

        tx_df = pd.DataFrame(tx_features)
        if tx_df.empty or y_tx is None:
            seq_p = seq_r = seq_f1 = np.nan
        else:
            tx_df["y_tx"] = tx_df[seq_id].map(y_tx).fillna(0).astype(int)

            X_tx = tx_df.drop(columns=["y_tx"], errors="ignore")
            y_txv = tx_df["y_tx"]

            num_cols_tx = [c for c in X_tx.columns if c != seq_id and pd.api.types.is_numeric_dtype(X_tx[c])]
            cat_cols_tx = [c for c in X_tx.columns if c == seq_id or c not in num_cols_tx]

            pre_tx = ColumnTransformer(
                transformers=[
                    ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols_tx),
                    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols_tx),
                ],
                remainder="drop"
            )

            if y_txv.nunique() < 2:
                seq_p = seq_r = seq_f1 = np.nan
            else:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_tx, y_txv, test_size=test_size, random_state=random_state, stratify=y_txv
                )
                model_tx = Pipeline([("pre", pre_tx), ("clf", LogisticRegression(max_iter=3000))])
                model_tx.fit(X_tr, y_tr)
                pred = model_tx.predict(X_te)
                seq_p, seq_r, seq_f1 = clf_metrics(y_te, pred)

    results = {
        "classification_failure": {"precision": failure_p, "recall": failure_r, "f1": failure_f1},
        "classification_anomaly": {"precision": anomaly_p, "recall": anomaly_r, "f1": anomaly_f1},
        "regression_latency": {"mae": lat_mae, "rmse": lat_rmse, "r2": lat_r2},
        "sequence_tx_failure": {"precision": float(seq_p), "recall": float(seq_r), "f1": float(seq_f1), "n_prefix": n_prefix, "seq_id": seq_id},
    }

    print("=== Classification (Event-level) ===")
    print(f"Failure   P/R/F1: {failure_p:.2%} / {failure_r:.2%} / {failure_f1:.2%}")
    print(f"Anomaly   P/R/F1: {anomaly_p:.2%} / {anomaly_r:.2%} / {anomaly_f1:.2%}")
    print("\n=== Regression (Event-level) ===")
    print(f"Latency   MAE/RMSE/R2: {lat_mae:.3f} / {lat_rmse:.3f} / {lat_r2:.3f}")
    print("\n=== Sequence (Transaction-level, prefix features) ===")
    if np.isnan(seq_p):
        print("Tx Failure P/R/F1: N/A (brak kolumny ID sekwencji albo brak obu klas)")
    else:
        print(f"Tx Failure P/R/F1: {seq_p:.2%} / {seq_r:.2%} / {seq_f1:.2%} (prefix={n_prefix}, id={seq_id})")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n=== Czas wykonania ===")
    print(f"Wykonanie trwało: {elapsed_time:.2f} sekund")

    return results
