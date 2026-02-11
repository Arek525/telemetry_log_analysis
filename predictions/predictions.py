import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _clf_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(p), float(r), float(f1)


def _quality_from_f1(f1_value):
    if np.isnan(f1_value):
        return "brak oceny"
    if f1_value >= 0.9:
        return "bardzo dobra"
    if f1_value >= 0.75:
        return "dobra"
    if f1_value >= 0.6:
        return "umiarkowana"
    return "slaba"


def _quality_from_r2(r2_value):
    if np.isnan(r2_value):
        return "brak oceny"
    if r2_value >= 0.7:
        return "bardzo dobra"
    if r2_value >= 0.5:
        return "dobra"
    if r2_value >= 0.3:
        return "umiarkowana"
    return "slaba"


def _safe_split(
    X,
    y,
    test_size,
    random_state,
    groups=None,
):
    y_series = pd.Series(y)
    can_stratify = y_series.nunique(dropna=False) >= 2

    if groups is None or pd.Series(groups).nunique(dropna=False) < 2:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if can_stratify else None,
        )

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_tr = X.iloc[train_idx]
    X_te = X.iloc[test_idx]
    y_tr = y.iloc[train_idx]
    y_te = y.iloc[test_idx]

    # Group split can occasionally produce a single class in train/test.
    if can_stratify and (y_tr.nunique(dropna=False) < 2 or y_te.nunique(dropna=False) < 2):
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    return X_tr, X_te, y_tr, y_te


def _event_preprocessor(X):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def predictions(
    data,
    seq_id="CorrelationId",
    n_prefix=3,
    test_size=0.2,
    random_state=42,
    max_event_rows=None,
    max_sequence_transactions=None,
):
    df = data.copy()

    if "AttributesJson" in df.columns:
        raise ValueError(
            "predictions() oczekuje dataframe po parsowaniu AttributesJson. "
            "Najpierw wykonaj json.loads + pd.json_normalize w main.py."
        )

    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
        df["TimestampHour"] = ts.dt.hour
        df["TimestampDayOfWeek"] = ts.dt.dayofweek

    for c in [
        "EventCode",
        "LatencyMs",
        "IsAnomaly",
        "CpuUsage",
        "LocalQps",
        "Retries",
        "DiskQueueLength",
        "NetworkErrors",
        "MemoryUsageMb",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "IsAnomaly" in df.columns:
        df["IsAnomaly"] = df["IsAnomaly"].fillna(0).astype(int)
    else:
        df["IsAnomaly"] = 0

    if "EventCode" not in df.columns:
        raise ValueError("Brak kolumny EventCode")
    if "LatencyMs" not in df.columns:
        raise ValueError("Brak kolumny LatencyMs")

    leak_cols = [
        c
        for c in df.columns
        if c.lower()
        in {
            "isspikebylatency",
            "isspikebycpu",
            "isspikebyqps",
            "spikewindow",
            "failurewindow",
            "trendwindow",
        }
    ]
    df = df.drop(columns=leak_cols, errors="ignore")

    df["is_failure"] = df["EventCode"].isin([500, 501]).astype(int)
    df["is_anomaly_target"] = (
        (df["IsAnomaly"] == 1) | (df["EventCode"].isin([998, 999]))
    ).astype(int)

    # Keep all positives and sample negatives for fast runtime.
    df_event = df
    if max_event_rows is not None and len(df) > max_event_rows:
        pos_mask = (df["is_failure"] == 1) | (df["is_anomaly_target"] == 1)
        pos_df = df[pos_mask]
        neg_df = df[~pos_mask]
        remaining = max(0, max_event_rows - len(pos_df))
        if remaining > 0:
            neg_df = neg_df.sample(
                n=min(remaining, len(neg_df)), random_state=random_state
            )
            df_event = pd.concat([pos_df, neg_df], axis=0)
        else:
            df_event = pos_df.sample(n=max_event_rows, random_state=random_state)
        df_event = df_event.sample(frac=1.0, random_state=random_state).reset_index(
            drop=True
        )

    excluded_feature_cols = {
        "EventCode",
        "IsAnomaly",
        "is_failure",
        "is_anomaly_target",
        "LatencyMs",
        "TransactionId",
        "CorrelationId",
        "Timestamp",
        "Description",
    }

    X_event = df_event.drop(
        columns=[c for c in excluded_feature_cols if c in df_event.columns],
        errors="ignore",
    ).copy()

    # Drop very high-cardinality categoricals to keep OHE compact and fast.
    cat_cols = [c for c in X_event.columns if not pd.api.types.is_numeric_dtype(X_event[c])]
    drop_high_card = []
    for c in cat_cols:
        if X_event[c].nunique(dropna=True) > 100:
            drop_high_card.append(c)
    X_event = X_event.drop(columns=drop_high_card, errors="ignore")

    y_failure = df_event["is_failure"]
    y_anomaly = df_event["is_anomaly_target"]
    y_latency = df_event["LatencyMs"]

    if seq_id not in df_event.columns:
        alt_id = "TransactionId" if "TransactionId" in df_event.columns else None
        effective_seq_id = alt_id
    else:
        effective_seq_id = seq_id

    groups = (
        df_event[effective_seq_id].fillna("__na__").astype(str)
        if effective_seq_id is not None
        else None
    )

    X_tr, X_te, y_fail_tr, y_fail_te = _safe_split(
        X_event,
        y_failure,
        test_size=test_size,
        random_state=random_state,
        groups=groups,
    )
    y_anom_tr = y_anomaly.loc[y_fail_tr.index]
    y_anom_te = y_anomaly.loc[y_fail_te.index]
    y_lat_tr = y_latency.loc[y_fail_tr.index]
    y_lat_te = y_latency.loc[y_fail_te.index]

    pre = _event_preprocessor(X_event)
    X_tr_tf = pre.fit_transform(X_tr)
    X_te_tf = pre.transform(X_te)

    failure_p = failure_r = failure_f1 = np.nan
    if y_fail_tr.nunique(dropna=False) >= 2 and y_fail_te.nunique(dropna=False) >= 1:
        clf_failure = LogisticRegression(
            max_iter=400, solver="liblinear", class_weight="balanced"
        )
        clf_failure.fit(X_tr_tf, y_fail_tr)
        pred_failure = clf_failure.predict(X_te_tf)
        failure_p, failure_r, failure_f1 = _clf_metrics(y_fail_te, pred_failure)

    anomaly_p = anomaly_r = anomaly_f1 = np.nan
    if y_anom_tr.nunique(dropna=False) >= 2 and y_anom_te.nunique(dropna=False) >= 1:
        clf_anomaly = LogisticRegression(
            max_iter=400, solver="liblinear", class_weight="balanced"
        )
        clf_anomaly.fit(X_tr_tf, y_anom_tr)
        pred_anomaly = clf_anomaly.predict(X_te_tf)
        anomaly_p, anomaly_r, anomaly_f1 = _clf_metrics(y_anom_te, pred_anomaly)

    lat_train_mask = y_lat_tr.notna()
    lat_test_mask = y_lat_te.notna()
    lat_mae = lat_rmse = lat_r2 = np.nan
    lat_baseline_mae = lat_baseline_rmse = np.nan
    lat_vs_baseline_mae_pct = np.nan
    if lat_train_mask.sum() >= 5 and lat_test_mask.sum() >= 5:
        reg = Ridge(alpha=1.0)
        reg.fit(X_tr_tf[lat_train_mask.values], y_lat_tr[lat_train_mask])
        pred_latency = reg.predict(X_te_tf[lat_test_mask.values])
        lat_mae = float(mean_absolute_error(y_lat_te[lat_test_mask], pred_latency))
        lat_rmse = float(
            np.sqrt(mean_squared_error(y_lat_te[lat_test_mask], pred_latency))
        )
        lat_r2 = float(r2_score(y_lat_te[lat_test_mask], pred_latency))

        baseline_value = float(y_lat_tr[lat_train_mask].median())
        baseline_pred = np.full(lat_test_mask.sum(), baseline_value, dtype=float)
        lat_baseline_mae = float(
            mean_absolute_error(y_lat_te[lat_test_mask], baseline_pred)
        )
        lat_baseline_rmse = float(
            np.sqrt(mean_squared_error(y_lat_te[lat_test_mask], baseline_pred))
        )
        if lat_baseline_mae > 0:
            lat_vs_baseline_mae_pct = float(
                (lat_baseline_mae - lat_mae) / lat_baseline_mae * 100.0
            )

    seq_p = seq_r = seq_f1 = np.nan
    used_seq_id = effective_seq_id if effective_seq_id is not None else seq_id
    seq_tx_count = 0

    if effective_seq_id is not None and effective_seq_id in df.columns:
        seq_df = df.copy()
        unique_txs = seq_df[effective_seq_id].dropna().unique()
        if (
            max_sequence_transactions is not None
            and len(unique_txs) > max_sequence_transactions
        ):
            rng = np.random.default_rng(random_state)
            chosen = rng.choice(unique_txs, size=max_sequence_transactions, replace=False)
            seq_df = seq_df[seq_df[effective_seq_id].isin(chosen)].copy()

        if "Timestamp" in seq_df.columns:
            seq_df["_ts"] = pd.to_datetime(seq_df["Timestamp"], errors="coerce", utc=True)
            seq_df = seq_df.sort_values([effective_seq_id, "_ts"])
        else:
            seq_df = seq_df.sort_values([effective_seq_id])

        seq_tx_count = int(seq_df[effective_seq_id].nunique(dropna=True))
        tx_has_failure = seq_df["EventCode"].isin([500, 501, 998, 999]).groupby(
            seq_df[effective_seq_id]
        ).max()
        tx_has_anomaly = (seq_df["IsAnomaly"] == 1).groupby(
            seq_df[effective_seq_id]
        ).max()
        y_tx = (tx_has_failure | tx_has_anomaly).astype(int)

        seq_prefix = seq_df.groupby(effective_seq_id, sort=False).head(n_prefix).copy()
        agg_map = {"EventCode": ["count"], "SourceSystem": ["nunique"], "Scenario": ["nunique"]}
        for c in [
            "CpuUsage",
            "LocalQps",
            "Retries",
            "DiskQueueLength",
            "NetworkErrors",
            "MemoryUsageMb",
            "LatencyMs",
        ]:
            if c in seq_prefix.columns:
                agg_map[c] = ["mean", "max"]

        tx_df = seq_prefix.groupby(effective_seq_id).agg(agg_map)
        tx_df.columns = ["_".join(col) for col in tx_df.columns]
        tx_df = tx_df.rename(
            columns={
                "EventCode_count": "events_in_prefix",
                "SourceSystem_nunique": "unique_services",
                "Scenario_nunique": "unique_scenarios",
            }
        )
        tx_df = tx_df.reset_index()
        tx_df["y_tx"] = tx_df[effective_seq_id].map(y_tx).fillna(0).astype(int)

        if tx_df["y_tx"].nunique() >= 2 and len(tx_df) >= 20:
            X_tx = tx_df.drop(columns=["y_tx", effective_seq_id], errors="ignore")
            y_txv = tx_df["y_tx"]

            X_tr_tx, X_te_tx, y_tr_tx, y_te_tx = train_test_split(
                X_tx,
                y_txv,
                test_size=test_size,
                random_state=random_state,
                stratify=y_txv,
            )

            pre_tx = _event_preprocessor(X_tx)
            model_tx = Pipeline(
                [
                    ("pre", pre_tx),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=400,
                            solver="liblinear",
                            class_weight="balanced",
                        ),
                    ),
                ]
            )
            model_tx.fit(X_tr_tx, y_tr_tx)
            pred_tx = model_tx.predict(X_te_tx)
            seq_p, seq_r, seq_f1 = _clf_metrics(y_te_tx, pred_tx)

    results = {
        "classification_failure": {
            "precision": failure_p,
            "recall": failure_r,
            "f1": failure_f1,
        },
        "classification_anomaly": {
            "precision": anomaly_p,
            "recall": anomaly_r,
            "f1": anomaly_f1,
        },
        "regression_latency": {"mae": lat_mae, "rmse": lat_rmse, "r2": lat_r2},
        "sequence_tx_failure": {
            "precision": float(seq_p),
            "recall": float(seq_r),
            "f1": float(seq_f1),
            "n_prefix": n_prefix,
            "seq_id": used_seq_id,
            "n_transactions": seq_tx_count,
        },
        "runtime_config": {
            "event_rows_used": int(len(df_event)),
            "max_event_rows": max_event_rows,
            "max_sequence_transactions": max_sequence_transactions,
        },
    }
    results["regression_latency"]["baseline_mae"] = lat_baseline_mae
    results["regression_latency"]["baseline_rmse"] = lat_baseline_rmse
    results["regression_latency"]["mae_improvement_vs_baseline_pct"] = lat_vs_baseline_mae_pct

    print("=== Raport predykcji zdarzen ===")
    print("Konfiguracja i zakres danych:")
    print(
        f"- Liczba eventow uzytych do modeli event-level: {len(df_event):,} / {len(df):,}"
    )
    print(f"- Liczba transakcji uzytych do modelu sekwencyjnego: {seq_tx_count:,}")
    print(f"- Dlugosc prefiksu sekwencji (n_prefix): {n_prefix}")
    if max_event_rows is None and max_sequence_transactions is None:
        print("- Tryb danych: pelny zbior (bez samplingu)")
    else:
        print(
            "- Tryb danych: sampling aktywny "
            f"(max_event_rows={max_event_rows}, "
            f"max_sequence_transactions={max_sequence_transactions})"
        )

    print("\n=== Klasyfikacja event-level: Failure ===")
    if np.isnan(failure_p):
        print("Brak oceny: za malo klas po podziale train/test.")
    else:
        print(
            f"Precision (Precyzja): {failure_p:.2%} -> "
            "jaki procent wykrytych awarii byl poprawny."
        )
        print(
            f"Recall (Czulosc): {failure_r:.2%} -> "
            "jaki procent wszystkich awarii zostal wykryty."
        )
        print(
            f"F1-score: {failure_f1:.2%} -> "
            f"zbalansowana jakosc (ocena: {_quality_from_f1(failure_f1)})."
        )

    print("\n=== Klasyfikacja event-level: Anomaly ===")
    if np.isnan(anomaly_p):
        print("Brak oceny: za malo klas po podziale train/test.")
    else:
        print(
            f"Precision (Precyzja): {anomaly_p:.2%} -> "
            "jaki procent wykrytych anomalii byl poprawny."
        )
        print(
            f"Recall (Czulosc): {anomaly_r:.2%} -> "
            "jaki procent wszystkich anomalii zostal wykryty."
        )
        print(
            f"F1-score: {anomaly_f1:.2%} -> "
            f"zbalansowana jakosc (ocena: {_quality_from_f1(anomaly_f1)})."
        )

    print("\n=== Regresja event-level: LatencyMs ===")
    if np.isnan(lat_mae):
        print("Brak oceny: za malo danych po podziale train/test.")
    else:
        print(
            f"MAE (Mean Absolute Error): {lat_mae:.3f} ms -> "
            "sredni blad bezwzgledny predykcji opoznienia."
        )
        print(
            f"RMSE (Root Mean Squared Error): {lat_rmse:.3f} ms -> "
            "blad mocniej karzacy duze pomylki."
        )
        print(
            f"R2 (Wspolczynnik determinacji): {lat_r2:.3f} -> "
            f"jakosc dopasowania (ocena: {_quality_from_r2(lat_r2)})."
        )
        if not np.isnan(lat_baseline_mae):
            print(
                f"Porownanie z baseline (mediana z train): "
                f"MAE modelu {lat_mae:.3f} ms vs MAE baseline {lat_baseline_mae:.3f} ms "
                f"(poprawa: {lat_vs_baseline_mae_pct:.2f}%)."
            )

    print("\n=== Predykcja sekwencyjna transaction-level ===")
    if np.isnan(seq_p):
        print("Brak oceny: brak ID sekwencji albo brak obu klas.")
    else:
        print(
            f"Precision (Precyzja): {seq_p:.2%} -> "
            "jaki procent alertow o awarii transakcji byl poprawny."
        )
        print(
            f"Recall (Czulosc): {seq_r:.2%} -> "
            "jaki procent transakcji problemowych wykryto juz po prefiksie."
        )
        print(
            f"F1-score: {seq_f1:.2%} -> "
            f"zbalansowana jakosc (ocena: {_quality_from_f1(seq_f1)}). "
            f"Uzyte ID sekwencji: {used_seq_id}."
        )

    recommendations = []
    if max_event_rows is not None or max_sequence_transactions is not None:
        recommendations.append(
            "Dla finalnej oceny porownaj wynik z uruchomieniem bez samplingu "
            "(max_event_rows=None, max_sequence_transactions=None)."
        )
    if np.isnan(failure_f1) or np.isnan(anomaly_f1):
        recommendations.append(
            "W klasyfikacji pojawil sie brak jednej klasy po podziale; "
            "zwieksz dane lub ustaw inny random_state/test_size."
        )
    else:
        if failure_r < 0.9 or anomaly_r < 0.9:
            recommendations.append(
                "Niski recall oznacza utracone incydenty; rozważ zwiekszenie recall "
                "kosztem precision (np. prog decyzyjny, class_weight)."
            )
        if failure_p < 0.85 or anomaly_p < 0.85:
            recommendations.append(
                "Niska precision oznacza wiele falszywych alarmow; "
                "warto podniesc prog alarmu i ograniczyc cechy szumowe."
            )
    if not np.isnan(lat_r2):
        if lat_r2 < 0.3:
            recommendations.append(
                "Regresja ma niska jakosc (R2<0.3); dodaj cechy czasowe i interakcje metryk."
            )
        if not np.isnan(lat_vs_baseline_mae_pct) and lat_vs_baseline_mae_pct < 5:
            recommendations.append(
                "Model regresji jest niewiele lepszy od baseline; "
                "sprawdz inne modele lub transformacje celu (np. log LatencyMs)."
            )
    if not np.isnan(seq_f1):
        if seq_f1 < 0.7:
            recommendations.append(
                "Predykcja sekwencyjna jest umiarkowana; "
                "zwieksz n_prefix albo dodaj cechy trendu w prefiksie."
            )
        if seq_r < 0.8:
            recommendations.append(
                "W sekwencji recall jest niski; model moze za pozno wykrywac awarie."
            )

    if not recommendations:
        recommendations.append(
            "Wyniki sa stabilne; kolejnym krokiem jest walidacja czasowa "
            "na nowszym okresie logow."
        )

    print("\n=== Rekomendacje na podstawie wyniku ===")
    for idx, line in enumerate(recommendations, start=1):
        print(f"{idx}. {line}")

    return results
