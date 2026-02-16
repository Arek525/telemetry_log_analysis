# Telemetry Log Analysis (Telemetry Data Engineering Project)

This repository contains a team project developed with my university classmates for the course **Telemetry Data Engineering**, delivered in collaboration with Dynatrace.

The goal is to analyze large, structured application logs from distributed services, detect abnormal behavior, and investigate probable root causes of incidents.

## Project Context

- Course topic: log and telemetry analysis in distributed systems
- Team setup: collaborative student project
- Data source: synthetic but realistic logs generated for analysis scenarios
- Focus: practical incident detection and diagnosis (spikes, anomalies, failures, propagation)

## What We Analyze

The logs represent multi-service business flows (for example: `AuthService`, `ApiGateway`, `OrderService`, `PaymentService`) and include:

- service and correlation context (`SourceSystem`, `TransactionId`, `CorrelationId`)
- event metadata (`Priority`, `EventKind`, `EventCode`, `Scenario`)
- performance and system metrics (`LatencyMs`, `CpuUsage`, `MemoryUsageMb`, `DiskQueueLength`, `LocalQps`, `NetworkErrors`, `Retries`)
- flags/windows for synthetic incident patterns (`SpikeWindow`, `FailureWindow`, `TrendWindow`)

## Main Analysis Modules

- `basic_info/index.py`
  - dataset-level statistics
  - distributions of key categorical fields
  - histograms for latency, CPU, QPS, and disk queue length

- `spikes/spike_detector.py`
  - heuristic spike detection for latency/CPU/QPS
  - per-service spike summaries
  - precision/recall/F1 against available ground-truth labels

- `failure_detection/index.py`
  - failure candidate scoring based on priority + metric thresholds
  - failure window detection from 5xx density in time windows
  - confusion-matrix evaluation and propagation summary by `CorrelationId`

- `anomaly_detection/index.py`
  - Isolation Forest anomaly detection over normalized numeric features
  - adaptive contamination estimate from 5xx base rate
  - anomaly metrics (precision/recall/F1) when labels are present
  - top anomalous scenarios and human-readable recommendations

- `root_cause_analysis/index.py`
  - correlation-level timeline reconstruction
  - first problematic service identification
  - victim/impact path analysis
  - pre-failure assessment and watch-list recommendations

- `binning_var_entropy/index.py`
  - metric binning
  - variance/entropy/balance-ratio comparisons across normal vs spike/failure/trend windows

## Repository Structure

```text
.
├── main.py
├── basic_info/
├── spikes/
├── failure_detection/
├── anomaly_detection/
├── root_cause_analysis/
├── binning_var_entropy/
├── SyntheticLogGenerator/      # .NET generator for realistic telemetry logs
└── info/                       # course notes/config snippets
```

## Data Generation

`SyntheticLogGenerator/` is a C#/.NET tool that generates large telemetry datasets (`logs.csv`) with controlled rates of:

- spike windows
- failure windows
- trend windows
- anomalies

It can also save generation statistics for validation and benchmarking.

## Setup

### 1. Python environment

Install required packages:

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 2. (Optional) Build/run log generator

From `SyntheticLogGenerator/`:

```bash
dotnet build
dotnet run
```

This produces `logs.csv` (path configured in `SyntheticLogGenerator/config.json`).

## Running the Analysis

The main entry point is:

```bash
python main.py
```

`main.py` loads `logs.csv`, expands `AttributesJson`, and runs selected analysis modules.  
You can enable/disable specific steps in `main.py` depending on the experiment.

## Typical Investigation Flow

1. Generate or load telemetry logs.
2. Compute baseline stats and metric distributions.
3. Detect spikes and failure candidates.
4. Run anomaly detection.
5. Perform root-cause analysis for suspicious correlations.
6. Compare behavioral patterns (binning/variance/entropy) between normal and incident windows.

## Why This Project

This project demonstrates how to combine classic statistical analysis, heuristic detection, and ML-based anomaly detection in one practical observability workflow.  
It reflects the type of reasoning used in production monitoring platforms and helps us practice incident-oriented thinking on realistic telemetry data.
