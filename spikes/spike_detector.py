import datetime
import pandas as pd
spikes_by_latency = set()
spikes_by_cpu = set()
spikes_by_qps = set()
services = {}
metrics = {
  "lat": {"tp":0,"fp":0,"tn":0,"fn":0},
  "cpu": {"tp":0,"fp":0,"tn":0,"fn":0},
  "qps": {"tp":0,"fp":0,"tn":0,"fn":0},
}
def find_timeframe(data, center_ts, window_minutes=5):
    delta = pd.Timedelta(minutes=window_minutes)
    return data.loc[center_ts - delta : center_ts]

def compute_scores(m):
    tp = m["tp"]
    fp = m["fp"]
    fn = m["fn"]

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return precision, recall, f1


def is_spike(data):
    data = data.copy()
    data["Timestamp"] = pd.to_datetime(
        data["Timestamp"],
        format="ISO8601",
        errors="coerce")
    data = data.dropna(subset=["Timestamp"])
    data = data.set_index("Timestamp").sort_index()
    is_spike_by_latency(data)
    is_spike_by_cpu(data)
    is_spike_by_qps(data)
    print('\n=========================================')
    print("SPIKES BY SERVICE:")
    for service, counts in services.items():
        print(f"Service: {service}, Total Spikes: {counts['total_spikes']}, Latency Spikes: {counts['latency']}, CPU Spikes: {counts['cpu']}, QPS Spikes: {counts['qps']}")
    print("\n===============================\n")
    print("OVERALL SPIKES:\n")
    print(f"Overall spikes detected by Latency: {len(spikes_by_latency)}")
    print(f"Overall spikes detected by CPU: {len(spikes_by_cpu)}")
    print(f"Overall spikes detected by QPS: {len(spikes_by_qps)}")
    print("Total (sum):", len(spikes_by_latency) + len(spikes_by_cpu) + len(spikes_by_qps))
    print("\n===============================\n")
    print("PRECISION / RECALL / F1:")
    for key, name in [("lat", "Latency"), ("cpu", "CPU"), ("qps", "QPS")]:
        p, r, f1 = compute_scores(metrics[key])
        print(f"{name}:")
        print(f"  Precision: {p:.2%}")
        print(f"  Recall:    {r:.2%}")
        print(f"  F1-score:  {f1:.2%}")
        print()
    print('\n=========================================\n')
    return services

def is_spike_by_latency(data,k=4):
    m = metrics["lat"]
    median_latency = data["LatencyMs"].median()
    average_latency = data["LatencyMs"].mean()
    standard_deviation_latency = data["LatencyMs"].std()
    for row in data.itertuples():
        if(row.SourceSystem not in services):
            services[row.SourceSystem] = {"total_spikes": 0, "latency": 0, "cpu": 0, "qps": 0}
        if(row.LatencyMs > k * median_latency or row.LatencyMs >  average_latency +  k * standard_deviation_latency):
            # print(f"Spike detected by Latency at {row.Index} with Latency={row.LatencyMs} (median={median_latency}, average={average_latency})")
            if(row.IsSpikeByLatency):
                m["tp"] += 1
            else:
                m["fp"] += 1
            spikes_by_latency.add(row.Index)
            services[row.SourceSystem]["total_spikes"] += 1
            services[row.SourceSystem]["latency"] += 1
        elif (not row.IsSpikeByLatency):
            m["tn"] += 1
        else:
            m["fn"] += 1

def is_spike_by_cpu(data,k=2):
    m = metrics["cpu"]
    median_cpu = data["CpuUsage"].median()
    average_cpu = data["CpuUsage"].mean()
    standard_deviation_cpu = data["CpuUsage"].std()
    for row in data.itertuples():
        if(row.SourceSystem not in services):
            services[row.SourceSystem] = {"total_spikes": 0, "latency": 0, "cpu": 0, "qps": 0}
        if(row.CpuUsage > k * median_cpu or row.CpuUsage > average_cpu + k * standard_deviation_cpu):
            # print("CPU: ",row.CpuUsage, k * median_cpu, average_cpu + k * standard_deviation_cpu)
            # print(f"Detected spike by CPU: {k * median_cpu}, {average_cpu + k * standard_deviation_cpu}")
            # print(f"Spike detected by CPU at {row.Index} with CpuUsage={row.CpuUsage} (median={median_cpu}, average={average_cpu})")
            if(row.IsSpikeByCpu):
                m["tp"] += 1
            else:
                m["fp"] += 1
            spikes_by_cpu.add(row.Index)
            services[row.SourceSystem]["total_spikes"] += 1
            services[row.SourceSystem]["cpu"] += 1
        else:
            if(not row.IsSpikeByCpu):
                m["tn"] += 1
            else:
                m["fn"] += 1

def is_spike_by_qps(data,k=3):
    m = metrics["qps"]
    median_qps = data["LocalQps"].median()
    standard_deviation_qps = data["LocalQps"].std()
    for row in data.itertuples():
        if(row.SourceSystem not in services):
            services[row.SourceSystem] = {"total_spikes": 0, "latency": 0, "cpu": 0, "qps": 0}
        if(row.LocalQps > k * median_qps or row.LocalQps > median_qps + k * standard_deviation_qps):
            # print("QPS: ",row.LocalQps, k * median_qps, median_qps + k * standard_deviation_qps)
            # print(f"Detected spike by QPS: {k * median_qps}, {median_qps + k * standard_deviation_qps}")
            # print(f"Spike detected by Local QPS at {row.Index} with LocalQps={row.LocalQps} (median={median_qps}, average={average_qps})")
            if(row.IsSpikeByQps):
                m["tp"] += 1
            else:
                m["fp"] += 1
            spikes_by_qps.add(row.Index)
            services[row.SourceSystem]["total_spikes"] += 1
            services[row.SourceSystem]["qps"] += 1
        else:
            if(not row.IsSpikeByQps):
                m["tn"] += 1
            else:
                m["fn"] += 1