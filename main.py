import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from basic_info.index import basic_info
from binning_var_entropy.index import b_v_e
from anomaly_detection.index import anomaly_detection
from failure_detection.index import failure_detection
from root_cause_analysis.index import root_cause_analysis
from spikes.spike_detector import is_spike
import Trends.trends_analytics as trends
import clustering.kmeans_analytics as kmeans
from predictions.predictions import predictions
import time

start = time.perf_counter()

base_dir = Path(__file__).resolve().parent
plots_dir = base_dir / "outputs" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

candidate_paths = [
    base_dir / "logs.csv",
    base_dir / "SyntheticLogGenerator" / "logs.csv",
]

for csv_path in candidate_paths:
    if csv_path.exists():
        df = pd.read_csv(csv_path, sep=";")
        break
else:
    searched = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        f"Nie znaleziono logs.csv. Sprawdzone lokalizacje: {searched}"
    )

df["AttributesJson"] = df["AttributesJson"].apply(json.loads)

attributes_df = pd.json_normalize(df["AttributesJson"])

df = pd.concat(
    [df.drop(columns=["AttributesJson"]), attributes_df],
    axis=1
)

# podstawowe informacje
basic_info(df, plots_dir=str(plots_dir))

# binning wariancja entropia balance ratio
b_v_e(df)

# spike detection
is_spike(df)

# trends analysis
trends.run_analysis(df, output_dir=str(plots_dir))

# failure detection
failure_detection(df)

# detekcja anomalii
anomaly_detection(df)

# root cause analysis
root_cause_analysis(df)

# predykcja zdarzen
predictions(
    df,
    seq_id="CorrelationId",
    n_prefix=3,
)

# k-means clustering analysis
kmeans.run_analysis(df)

end = time.perf_counter()
print(f"Czas wykonania: {end - start:.3f} s")
