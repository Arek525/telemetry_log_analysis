import json
import numpy as np
import pandas as pd


LATENCY_THRESHOLD = 87.91 + 2 * 213.69
DISKQ_THRESHOLD = 14.08 + 3 * 20.8
NETERR_THRESHOLD = 1.59 + 3 * 2.94
CPU_THRESHOLD = 36.05 + 3 * 17.96

SCORE_THRESHOLD = 6.5


def is_failure_candidate_row(row, thr):
    event_kind = row.get("EventKind", row.get("EventType", row.get("Kind", None)))
    if isinstance(event_kind, str) and event_kind in ("Trace", "Audit", "Heartbeat"):
        return 0

    latency = to_number(row.get("LatencyMs", 0))
    retries = to_number(row.get("Retries", 0))
    diskq = to_number(row.get("DiskQueueLength", 0))
    neterr = to_number(row.get("NetworkErrors", 0))
    cpu = to_number(row.get("CpuUsage", 0))

    priority = row.get("Priority", "")
    priority_str = str(priority).upper()

    score = 0.0

    if priority_str in ("ERR", "ERROR", "CRIT", "CRITICAL"):
        score += 4

    if latency > thr["LAT"] * 2:
        score += 4
    elif latency > thr["LAT"] * 1.5:
        score += 2
    elif latency > thr["LAT"]:
        score += 1

    if diskq > thr["DISKQ"] * 3:
        score += 1.5
    elif diskq > thr["DISKQ"]:
        score += 1

    if neterr > thr["NETERR"]:
        score += 0.5
    if neterr > 5:
        score += 0.5

    if retries >= 3:
        score += 2
    elif retries > 0:
        score += 1

    if cpu > 95:
        score += 1
    elif cpu > thr["CPU"]:
        score += 0.5

    return int(score >= SCORE_THRESHOLD)



def failure_detection(df, stats_path=None):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  FAILURE DETECTION")
    print(sep)

    df2 = df.copy()

    for col in ("LatencyMs", "DiskQueueLength", "NetworkErrors", "CpuUsage", "Retries"):
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce").fillna(0)
        else:
            df2[col] = 0

    thr = {
        "LAT": float(df2["LatencyMs"].mean() + 2 * df2["LatencyMs"].std(ddof=0)),
        "DISKQ": float(df2["DiskQueueLength"].mean() + 3 * df2["DiskQueueLength"].std(ddof=0)),
        "NETERR": float(df2["NetworkErrors"].mean() + 3 * df2["NetworkErrors"].std(ddof=0)),
        "CPU": float(df2["CpuUsage"].mean() + 3 * df2["CpuUsage"].std(ddof=0)),
    }
    
    if "FailureWindow" not in df2.columns:
        time_col = pick_time_column(df2)
        if ("EventCode" in df2.columns) and time_col:
            df2[time_col] = pd.to_datetime(df2[time_col], errors="coerce")
            df2["_is5xx"] = df2["EventCode"].isin([500, 501])

            w = (
                df2.dropna(subset=[time_col])
                .set_index(time_col)
                .sort_index()
                .groupby(pd.Grouper(freq="5min"))["_is5xx"]
                .mean()
            )
            bad_windows = w[w >= 0.10].index

            df2["FailureWindow"] = df2[time_col].dt.floor("5min").isin(bad_windows).fillna(False)
            df2.drop(columns=["_is5xx"], inplace=True)
        else:
            df2["FailureWindow"] = False


    df2["FailureCandidate"] = df2.apply(lambda r: is_failure_candidate_row(r, thr), axis=1)

    candidates_count = int(df2["FailureCandidate"].sum())
    print("Znalezione failure candidates:", candidates_count)

    if "EventCode" in df2.columns:
        df2["IsRealFailureCode"] = df2["EventCode"].isin([500, 501]).astype(int)
    else:
        df2["IsRealFailureCode"] = 0

    df2["IsFailureWindow"] = df2["FailureWindow"].astype(int) if df2["FailureWindow"].dtype != int else df2["FailureWindow"]

    print("")
    analyze_confusion(
        df2["FailureCandidate"].astype(int),
        df2["IsRealFailureCode"].astype(int),
        title="ANALIZA NA BAZIE KODÓW (EventCode 500/501)"
    )

    print("")
    analyze_confusion(
        df2["FailureCandidate"].astype(int),
        df2["IsFailureWindow"].astype(int),
        title="ANALIZA NA BAZIE FLAGI FAILURE WINDOW"
    )

    if stats_path:
        print("")
        compare_with_stats_json(stats_path, df2, thr)

    print("")
    failure_propagation(df2)

    return df2


def analyze_confusion(pred, real, title):
    TP = int(((pred == 1) & (real == 1)).sum())
    TN = int(((pred == 0) & (real == 0)).sum())
    FP = int(((pred == 1) & (real == 0)).sum())
    FN = int(((pred == 0) & (real == 1)).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(title)
    print("True positives", TP)
    print("True negatives", TN)
    print("False positives", FP)
    print("False negatives", FN)
    print(f"precision: {precision:.3f} ({precision*100:.3f}%)")
    print(f"recall:    {recall:.3f} ({recall*100:.3f}%)")
    print(f"accuracy:  {accuracy:.3f} ({accuracy*100:.3f}%)")
    print(f"f1:        {f1:.3f} ({f1*100:.3f}%)")


def failure_propagation(df):
    required = ["CorrelationId", "SourceSystem", "FailureCandidate"]
    if not all(col in df.columns for col in required):
        print("PROPAGACJA: brak kolumn (CorrelationId/SourceSystem/FailureCandidate), pomijam.")
        return

    time_col = pick_time_column(df)

    if time_col:
        dfp = df.sort_values([ "CorrelationId", time_col ]).copy()
    else:
        dfp = df.copy()
        dfp["__Order"] = np.arange(len(dfp))
        dfp = dfp.sort_values([ "CorrelationId", "__Order" ])

    cand = dfp[dfp["FailureCandidate"] == 1]
    if len(cand) == 0:
        print("PROPAGACJA: brak failure candidates, pomijam.")
        return

    first_hits = (
        cand.groupby("CorrelationId", as_index=False)
        .head(1)[["CorrelationId", "SourceSystem"]]
        .rename(columns={"SourceSystem": "FirstFailSystem"})
    )

    dfm = dfp.merge(first_hits, on="CorrelationId", how="inner")

    summary = (
        dfm.groupby(["FirstFailSystem", "SourceSystem"], as_index=False)
        .agg(
            Events=("FailureCandidate", "size"),
            Candidates=("FailureCandidate", "sum")
        )
    )
    summary["CandidateRate"] = summary["Candidates"] / summary["Events"]
    summary = summary.sort_values(["FirstFailSystem", "CandidateRate", "Candidates"], ascending=[True, False, False])

    print("PROPAGACJA (CorrelationId): gdy pierwszy failure w systemie X, ile kandydatów w systemach Y w tej samej korelacji")
    print(summary.to_string(index=False))


def compare_with_stats_json(stats_path, df2, thr):
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
            total_events = int(stats.get("TotalEvents", 0)) if isinstance(stats, dict) else 0
            failure_events = int(stats.get("FailureEvents", 0)) if isinstance(stats, dict) else 0
    except Exception as e:
        print("Nie moge wczytac stats json:", stats_path)
        print("Blad:", str(e))
        return

    print("POROWNANIE ZE .stats.json:", stats_path)
    print("Progi użyte w regule (dynamiczne z danych):")
    print("Latency threshold:", thr["LAT"])
    print("DiskQueueLength threshold:", thr["DISKQ"])
    print("NetworkErrors threshold:", thr["NETERR"])
    print("CpuUsage threshold:", thr["CPU"])

    if isinstance(stats, dict):
        keys = ["LatencyMs", "DiskQueueLength", "NetworkErrors", "CpuUsage", "Retries"]
        found_any = False
        for k in keys:
            if k in stats:
                found_any = True
                print("")
                print("Statystyki dla:", k)
                try:
                    print(json.dumps(stats[k], ensure_ascii=False, indent=2))
                except Exception:
                    print(str(stats[k]))
        if not found_any:
            print("W stats json nie widze kluczy metryk (LatencyMs/DiskQueueLength/NetworkErrors/CpuUsage/Retries).")
    else:
        print("Stats json nie jest slownikiem, nie mam jak go sensownie porownac.")\
    
    detected_candidates = int(df2["FailureCandidate"].sum())
    detected_rate = detected_candidates / len(df2) if len(df2) else 0

    print("Wykryte FailureCandidate:", detected_candidates)
    print("FailureCandidateRate:", detected_rate)

    if total_events > 0 and failure_events > 0:
        expected_rate = failure_events / total_events
        print("Różnica rate (kandydaci - generator):", detected_rate - expected_rate)


def pick_time_column(df):
    candidates = ["Timestamp", "TimeStamp", "EventTime", "CreatedAt", "DateTime"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_number(x):
    try:
        if x is None:
            return 0
        if isinstance(x, bool):
            return int(x)
        return float(x)
    except Exception:
        return 0
