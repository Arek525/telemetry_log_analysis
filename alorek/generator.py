import csv
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

# ===================== CONFIG =====================

DEFAULT_CONFIG = {
    "output_path": "logs.csv",
    "target_size_mb": 1024,        # ZAMIENNIK GB (bezpieczne na laby)
    "seed": 123,
    "system_count": 5,
    "expected_spike_windows": 10,
    "expected_failure_windows": 5,
    "expected_trend_windows": 5,
    "anomaly_rate": 0.0005,
    "avg_inter_event_time": 0.5
}

SYSTEMS = [
    "AuthService",
    "ApiGateway",
    "OrderService",
    "PaymentService",
    "NotificationService",
    "InventoryService",
    "ReportingService",
]

USERS = [
    "alice", "bob", "carol", "dave", "eve",
    "student01", "student02", "qa_user", "service_account"
]

SCENARIOS = [
    "OrderPlacement", "UserLogin", "PasswordReset",
    "CheckoutWithCard", "CheckoutWithTransfer",
    "StockReservation", "EmailConfirmation",
    "RefundProcessing", "BulkImport", "ReportGeneration"
]

CSV_HEADER = [
    "Timestamp", "Priority", "User", "SourceSystem", "EventKind",
    "EventCode", "TransactionId", "CorrelationId",
    "Description", "LatencyMs", "IsAnomaly", "Scenario", "AttributesJson"
]

# ===================== HELPERS =====================

def load_config(path="config.json"):
    if Path(path).exists():
        with open(path) as f:
            cfg = json.load(f)
        return {**DEFAULT_CONFIG, **cfg}
    return DEFAULT_CONFIG


def create_windows(rnd, count, total_events, frac):
    windows = []
    length = max(10, int(total_events * frac))
    for _ in range(count):
        c = rnd.randint(0, total_events)
        s = max(0, c - length // 2)
        e = min(total_events, s + length)
        windows.append((s, e))
    return windows


def in_window(windows, idx):
    return any(s <= idx <= e for s, e in windows)


def build_metrics(rnd, idx, spike, failure, trend, anomaly):
    latency = rnd.randint(40, 70)
    if trend:
        latency += idx % 2000 // 40
    if spike:
        latency += rnd.randint(150, 400)
    if failure:
        latency += rnd.randint(500, 2000)
    if anomaly:
        latency = rnd.randint(2000, 8000)

    cpu = rnd.uniform(10, 55)
    if spike:
        cpu = rnd.uniform(80, 100)
    if anomaly:
        cpu = rnd.choice([rnd.uniform(0, 5), rnd.uniform(98, 100)])

    qps = rnd.uniform(10, 50)
    if spike:
        qps = rnd.uniform(100, 400)

    return {
        "LatencyMs": latency,
        "CpuUsage": round(cpu, 2),
        "LocalQps": round(qps, 2),
        "MemoryUsageMb": rnd.randint(200, 3000),
        "Retries": rnd.randint(0, 4) if failure else 0
    }

# ===================== MAIN =====================

def main():
    cfg = load_config()
    rnd = random.Random(cfg["seed"])

    systems = SYSTEMS[:cfg["system_count"]]
    target_bytes = cfg["target_size_mb"] * 1024 * 1024
    approx_events = target_bytes // 300

    spike_w = create_windows(rnd, cfg["expected_spike_windows"], approx_events, 0.005)
    fail_w = create_windows(rnd, cfg["expected_failure_windows"], approx_events, 0.002)
    trend_w = create_windows(rnd, cfg["expected_trend_windows"], approx_events, 0.02)

    now = datetime.utcnow() - timedelta(days=30)
    size = 0
    idx = 0

    counters = {
        "total": 0,
        "spike": 0,
        "failure": 0,
        "trend": 0,
        "anomaly": 0
    }

    with open(cfg["output_path"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(CSV_HEADER)

        while size < target_bytes:
            system = rnd.choice(systems)
            scenario = rnd.choice(SCENARIOS)
            user = rnd.choice(USERS)

            spike = in_window(spike_w, idx)
            failure = in_window(fail_w, idx)
            trend = in_window(trend_w, idx)
            anomaly = rnd.random() < cfg["anomaly_rate"]

            if spike: counters["spike"] += 1
            if failure: counters["failure"] += 1
            if trend: counters["trend"] += 1
            if anomaly: counters["anomaly"] += 1

            metrics = build_metrics(rnd, idx, spike, failure, trend, anomaly)

            row = [
                now.isoformat(),
                "info",
                user,
                system,
                "BusinessAction",
                500 if failure else 200,
                f"tx-{rnd.randint(100000,999999)}",
                f"corr-{rnd.randint(100000,999999)}",
                f"{scenario} on {system}",
                metrics["LatencyMs"],
                int(anomaly),
                scenario,
                json.dumps(metrics)
            ]

            writer.writerow(row)
            size += sum(len(str(x)) for x in row) + 20

            counters["total"] += 1
            idx += 1
            now += timedelta(seconds=cfg["avg_inter_event_time"] * rnd.uniform(0.3, 2.0))

    stats = {
        "generated_at": datetime.utcnow().isoformat(),
        "output": cfg["output_path"],
        "target_size_mb": cfg["target_size_mb"],
        "events": counters
    }

    with open(cfg["output_path"] + ".stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("✅ Generacja zakończona")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
