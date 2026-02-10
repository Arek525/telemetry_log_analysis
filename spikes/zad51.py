import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import Counter

CSV_PATH = "logs.csv"
CHUNKSIZE = 200_000   # możesz zmienić (np. 50_000 jeśli mało RAM)

def parse_attrs_series(s: pd.Series) -> pd.Series:
    # s: kolumna z AttributesJson
    def _one(x):
        if pd.isna(x):
            return {}
        try:
            x = str(x).strip()
            if not x.startswith("{"):
                x = "{" + x + "}"
            return json.loads(x)
        except Exception:
            return {}
    return s.apply(_one)

# --- Agregaty globalne ---
event_count = 0
txn_set = set()
corr_set = set()

priority_counter = Counter()
eventcode_counter = Counter()
sourcesystem_counter = Counter()
scenario_counter = Counter()

# histogramy: zbieramy wartości (jeśli gigantyczny plik, patrz komentarz niżej)
latency_vals = []
cpu_vals = []
qps_vals = []
diskq_vals = []

# trend w czasie (średnia per 1 min) dla Latency i QPS
# będziemy sumować i liczyć count na minutę
latency_sum_per_min = Counter()
latency_cnt_per_min = Counter()
qps_sum_per_min = Counter()
qps_cnt_per_min = Counter()

for chunk in pd.read_csv(
    CSV_PATH,
    sep=";",
    chunksize=CHUNKSIZE,
    parse_dates=["Timestamp"]
):
    event_count += len(chunk)

    # distinct IDs
    txn_set.update(chunk["TransactionId"].dropna().astype(str).unique())
    corr_set.update(chunk["CorrelationId"].dropna().astype(str).unique())

    # rozkłady
    priority_counter.update(chunk["Priority"].dropna().astype(str).tolist())
    eventcode_counter.update(chunk["EventCode"].dropna().astype(str).tolist())
    sourcesystem_counter.update(chunk["SourceSystem"].dropna().astype(str).tolist())
    scenario_counter.update(chunk["Scenario"].dropna().astype(str).tolist())

    # parse attributes -> normalize
    attrs = parse_attrs_series(chunk["AttributesJson"])
    attrs_df = pd.json_normalize(attrs)
    chunk = pd.concat([chunk, attrs_df], axis=1)

    # numeric conversions
    for col in ["LatencyMs", "CpuUsage", "LocalQps", "DiskQueueLength"]:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    # histogram values (dokładnie)
    if "LatencyMs" in chunk.columns:
        latency_vals.extend(chunk["LatencyMs"].dropna().tolist())
    if "CpuUsage" in chunk.columns:
        cpu_vals.extend(chunk["CpuUsage"].dropna().tolist())
    if "LocalQps" in chunk.columns:
        qps_vals.extend(chunk["LocalQps"].dropna().tolist())
    if "DiskQueueLength" in chunk.columns:
        diskq_vals.extend(chunk["DiskQueueLength"].dropna().tolist())

    # trend per 1 min (sum i count)
    # (unikamy plotowania punkt po punkcie)
    chunk["Minute"] = chunk["Timestamp"].dt.floor("min")

    if "LatencyMs" in chunk.columns:
        g = chunk.dropna(subset=["LatencyMs"]).groupby("Minute")["LatencyMs"].agg(["sum", "count"])
        for m, row in g.iterrows():
            latency_sum_per_min[m] += float(row["sum"])
            latency_cnt_per_min[m] += int(row["count"])

    if "LocalQps" in chunk.columns:
        g = chunk.dropna(subset=["LocalQps"]).groupby("Minute")["LocalQps"].agg(["sum", "count"])
        for m, row in g.iterrows():
            qps_sum_per_min[m] += float(row["sum"])
            qps_cnt_per_min[m] += int(row["count"])


# --- Wyniki liczników ---
print("Events:", event_count)
print("Distinct TransactionId:", len(txn_set))
print("Distinct CorrelationId:", len(corr_set))

# --- Rozkłady (top 20) ---
def print_top(counter: Counter, title: str, n=20):
    print("\n" + title)
    for k, v in counter.most_common(n):
        print(f"{k}: {v}")

print_top(priority_counter, "Priority distribution (top 20)")
print_top(eventcode_counter, "EventCode distribution (top 20)")
print_top(sourcesystem_counter, "SourceSystem distribution (top 20)")
print_top(scenario_counter, "Scenario distribution (top 20)")

# --- Histogramy (wymagane) ---
def hist_plot(values, title, xlabel, bins=60):
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.show()

hist_plot(latency_vals, "Histogram: LatencyMs", "Latency [ms]")
hist_plot(cpu_vals, "Histogram: CpuUsage", "CPU Usage [%]")
hist_plot(qps_vals, "Histogram: LocalQps", "QPS")
hist_plot(diskq_vals, "Histogram: DiskQueueLength", "Disk Queue Length")

# --- Trendy w czasie (opcjonalne, ale czytelne) ---
def build_series(sum_counter: Counter, cnt_counter: Counter):
    idx = sorted(cnt_counter.keys())
    vals = [(sum_counter[t] / cnt_counter[t]) if cnt_counter[t] else None for t in idx]
    return pd.Series(vals, index=pd.to_datetime(idx))

latency_min_series = build_series(latency_sum_per_min, latency_cnt_per_min)
qps_min_series = build_series(qps_sum_per_min, qps_cnt_per_min)

plt.figure()
plt.plot(latency_min_series.index, latency_min_series.values)
plt.title("Avg Latency per minute (all records)")
plt.xlabel("Time")
plt.ylabel("Latency [ms]")
plt.show()

plt.figure()
plt.plot(qps_min_series.index, qps_min_series.values)
plt.title("Avg QPS per minute (all records)")
plt.xlabel("Time")
plt.ylabel("QPS")
plt.show()
