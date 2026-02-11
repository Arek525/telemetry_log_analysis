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
    print("SPIKE RATES BY SERVICE:")
    for service, counts in services.items():
        total_events = counts["total_events"]
        if total_events > 0:
            spike_rate = counts["total_spikes"] / total_events
            print(f"Service: {service}, Spike Rate: {spike_rate:.2%}")
    print("\n===============================\n")
    print("OVERALL SPIKES:\n")
    print(f"Overall spikes detected by Latency: {len(spikes_by_latency)}")
    print(f"Overall spikes detected by CPU: {len(spikes_by_cpu)}")
    print(f"Overall spikes detected by QPS: {len(spikes_by_qps)}")
    print("\n===============================\n")
    print("POSITIVE/NEGATIVE RATES:\n")
    print(f"True Positives (TP): {metrics['lat']['tp']/(metrics['lat']['tp'] + metrics['lat']['fn'] + 1e-10):.2%} (Latency), {metrics['cpu']['tp']/(metrics['cpu']['tp'] + metrics['cpu']['fn'] + 1e-10):.2%} (CPU), {metrics['qps']['tp']/(metrics['qps']['tp'] + metrics['qps']['fn'] + 1e-10):.2%} (QPS)")
    print(f"False Positives (FP): {metrics['lat']['fp']/(metrics['lat']['fp'] + metrics['lat']['tn'] + 1e-10):.2%} (Latency), {metrics['cpu']['fp']/(metrics['cpu']['fp'] + metrics['cpu']['tn'] + 1e-10):.2%} (CPU), {metrics['qps']['fp']/(metrics['qps']['fp'] + metrics['qps']['tn'] + 1e-10):.2%} (QPS)")
    print(f"True Negatives (TN): {metrics['lat']['tn']/(metrics['lat']['tn'] + metrics['lat']['fp'] + 1e-10):.2%} (Latency), {metrics['cpu']['tn']/(metrics['cpu']['tn'] + metrics['cpu']['fp'] + 1e-10):.2%} (CPU), {metrics['qps']['tn']/(metrics['qps']['tn'] + metrics['qps']['fp'] + 1e-10):.2%} (QPS)")
    print(f"False Negatives (FN): {metrics['lat']['fn']/(metrics['lat']['fn'] + metrics['lat']['tp'] + 1e-10):.2%} (Latency), {metrics['cpu']['fn']/(metrics['cpu']['fn'] + metrics['cpu']['tp'] + 1e-10):.2%} (CPU), {metrics['qps']['fn']/(metrics['qps']['fn'] + metrics['qps']['tp'] + 1e-10):.2%} (QPS)")
    print('\n=========================================\n')
    return services

def is_spike_by_latency(data,k=3):
    m = metrics["lat"]
    # Grupowanie po serwisach
    service_groups = data.groupby("SourceSystem")
    service_metrics = {}
    
    # Obliczanie metryk dla każdego serwisu
    for service, group in service_groups:
        service_metrics[service] = {
            "median": group["LatencyMs"].median(),
            "average": group["LatencyMs"].mean(),
            "std": group["LatencyMs"].std()
        }
    
    for row in data.itertuples():
        if(row.SourceSystem not in services):
            services[row.SourceSystem] = {"total_spikes": 0, "latency": 0, "cpu": 0, "qps": 0,"total_events":0}
        services[row.SourceSystem]["total_events"] += 1
        
        # Używanie metryk specyficznych dla serwisu
        service_median = service_metrics[row.SourceSystem]["median"]
        service_average = service_metrics[row.SourceSystem]["average"]
        service_std = service_metrics[row.SourceSystem]["std"]
        
        if(row.LatencyMs > k * service_median or row.LatencyMs >  service_average +  k * service_std):
            # print(f"Spike detected by Latency at {row.Index} with Latency={row.LatencyMs} (median={service_median}, average={service_average})")
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
    # Grupowanie po serwisach
    service_groups = data.groupby("SourceSystem")
    service_metrics = {}
    
    # Obliczanie metryk dla każdego serwisu
    for service, group in service_groups:
        service_metrics[service] = {
            "median": group["CpuUsage"].median(),
            "average": group["CpuUsage"].mean(),
            "std": group["CpuUsage"].std()
        }
    
    for row in data.itertuples():
        if(row.SourceSystem not in services):
            services[row.SourceSystem] = {"total_spikes": 0, "latency": 0, "cpu": 0, "qps": 0,"total_events":0}
        services[row.SourceSystem]["total_events"] += 1
        
        # Używanie metryk specyficznych dla serwisu
        service_median = service_metrics[row.SourceSystem]["median"]
        service_average = service_metrics[row.SourceSystem]["average"]
        service_std = service_metrics[row.SourceSystem]["std"]
        
        if(row.CpuUsage > k * service_median or row.CpuUsage > service_average + k * service_std):
            # print("CPU: ",row.CpuUsage, k * service_median, service_average + k * service_std)
            # print(f"Detected spike by CPU: {k * service_median}, {service_average + k * service_std}")
            # print(f"Spike detected by CPU at {row.Index} with CpuUsage={row.CpuUsage} (median={service_median}, average={service_average})")
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
    # Grupowanie po serwisach
    service_groups = data.groupby("SourceSystem")
    service_metrics = {}
    
    # Obliczanie metryk dla każdego serwisu
    for service, group in service_groups:
        service_metrics[service] = {
            "median": group["LocalQps"].median(),
            "std": group["LocalQps"].std()
        }
    
    for row in data.itertuples():
        if(row.SourceSystem not in services):
            services[row.SourceSystem] = {"total_spikes": 0, "latency": 0, "cpu": 0, "qps": 0,"total_events":0}
        services[row.SourceSystem]["total_events"] += 1
        
        # Używanie metryk specyficznych dla serwisu
        service_median = service_metrics[row.SourceSystem]["median"]
        service_std = service_metrics[row.SourceSystem]["std"]
        
        if(row.LocalQps > k * service_median or row.LocalQps > service_median + k * service_std):
            # print("QPS: ",row.LocalQps, k * service_median, service_median + k * service_std)
            # print(f"Detected spike by QPS: {k * service_median}, {service_median + k * service_std}")
            # print(f"Spike detected by Local QPS at {row.Index} with LocalQps={row.LocalQps} (median={service_median})")
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