import datetime
import pandas as pd
spikes_by_latency = set()
spikes_by_cpu = set()
spikes_by_qps = set()
latency_spike = 0
cpu_spike = 0
qps_spike = 0
metrics = {
  "lat": {"tp":0,"fp":0,"tn":0,"fn":0},
  "cpu": {"tp":0,"fp":0,"tn":0,"fn":0},
  "qps": {"tp":0,"fp":0,"tn":0,"fn":0},
}
def find_timeframe(data, center_ts, window_minutes=5):
    delta = pd.Timedelta(minutes=window_minutes)
    return data.loc[center_ts - delta : center_ts]

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
    print("Spike detection results - moje:")
    print(f"Spikes detected by Latency: {len(spikes_by_latency)}")
    print(f"Spikes detected by CPU: {len(spikes_by_cpu)}")
    print(f"Spikes detected by QPS: {len(spikes_by_qps)}")
    print("Total (sum):", len(spikes_by_latency) + len(spikes_by_cpu) + len(spikes_by_qps))
    # print(f"True Positives (TP): Latency={metrics['lat']['tp']}, CPU={metrics['cpu']['tp']}, QPS={metrics['qps']['tp']}")
    # print(f"False Positives (FP): Latency={metrics['lat']['fp']}, CPU={metrics['cpu']['fp']}, QPS={metrics['qps']['fp']}")
    # print(f"True Negatives (TN): Latency={metrics['lat']['tn']}, CPU={metrics['cpu']['tn']}, QPS={metrics['qps']['tn']}")
    # print(f"False Negatives (FN): Latency={metrics['lat']['fn']}, CPU={metrics['cpu']['fn']}, QPS={metrics['qps']['fn']}")
    print(f"True Positives (TP): {metrics['lat']['tp']/(metrics['lat']['tp'] + metrics['lat']['fn'] + 1e-10):.2%} (Latency), {metrics['cpu']['tp']/(metrics['cpu']['tp'] + metrics['cpu']['fn'] + 1e-10):.2%} (CPU), {metrics['qps']['tp']/(metrics['qps']['tp'] + metrics['qps']['fn'] + 1e-10):.2%} (QPS)")
    print(f"False Positives (FP): {metrics['lat']['fp']/(metrics['lat']['fp'] + metrics['lat']['tn'] + 1e-10):.2%} (Latency), {metrics['cpu']['fp']/(metrics['cpu']['fp'] + metrics['cpu']['tn'] + 1e-10):.2%} (CPU), {metrics['qps']['fp']/(metrics['qps']['fp'] + metrics['qps']['tn'] + 1e-10):.2%} (QPS)")
    print(f"True Negatives (TN): {metrics['lat']['tn']/(metrics['lat']['tn'] + metrics['lat']['fp'] + 1e-10):.2%} (Latency), {metrics['cpu']['tn']/(metrics['cpu']['tn'] + metrics['cpu']['fp'] + 1e-10):.2%} (CPU), {metrics['qps']['tn']/(metrics['qps']['tn'] + metrics['qps']['fp'] + 1e-10):.2%} (QPS)")
    print(f"False Negatives (FN): {metrics['lat']['fn']/(metrics['lat']['fn'] + metrics['lat']['tp'] + 1e-10):.2%} (Latency), {metrics['cpu']['fn']/(metrics['cpu']['fn'] + metrics['cpu']['tp'] + 1e-10):.2%} (CPU), {metrics['qps']['fn']/(metrics['qps']['fn'] + metrics['qps']['tp'] + 1e-10):.2%} (QPS)")
    return spikes_by_latency, spikes_by_cpu, spikes_by_qps

def is_spike_by_latency(data,k=3):
    m = metrics["lat"]
    median_latency = data["LatencyMs"].median()
    average_latency = data["LatencyMs"].mean()
    standard_deviation_latency = data["LatencyMs"].std()
    for row in data.itertuples():
        if(row.LatencyMs > k * median_latency or row.LatencyMs >  average_latency +  k * standard_deviation_latency):
            # print(f"Spike detected by Latency at {row.Index} with Latency={row.LatencyMs} (median={median_latency}, average={average_latency})")
            if(row.IsSpikeByLatency):
                m["tp"] += 1
            else:
                m["fp"] += 1
            spikes_by_latency.add(row.Index)
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
        if(row.CpuUsage > k * median_cpu or row.CpuUsage > average_cpu + k * standard_deviation_cpu):
            # print(f"Spike detected by CPU at {row.Index} with CpuUsage={row.CpuUsage} (median={median_cpu}, average={average_cpu})")
            if(row.IsSpikeByCpu):
                m["tp"] += 1
            else:
                m["fp"] += 1
            spikes_by_cpu.add(row.Index)
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
        if(row.LocalQps > k * median_qps or row.LocalQps > median_qps + k * standard_deviation_qps):
            # print(f"Spike detected by Local QPS at {row.Index} with LocalQps={row.LocalQps} (median={median_qps}, average={average_qps})")
            if(row.IsSpikeByQps):
                m["tp"] += 1
            else:
                m["fp"] += 1
            spikes_by_qps.add(row.Index)
        else:
            if(not row.IsSpikeByQps):
                m["tn"] += 1
            else:
                m["fn"] += 1