import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from basic_info.index import basic_info
from binning_var_entropy.index import b_v_e
from spikes.spike_detector import is_spike
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
# pd.set_option("display.width", None)
# pd.set_option("display.max_colwidth", None)
df = pd.read_csv("logs.csv",sep=";")

df["AttributesJson"] = df["AttributesJson"].apply(json.loads)

attributes_df = pd.json_normalize(df["AttributesJson"])

df = pd.concat(
    [df.drop(columns=["AttributesJson"]), attributes_df],
    axis=1
)
# latency_spike = df["IsSpikeByLatency"].sum()
# cpu_spike = df["IsSpikeByCpu"].sum()
# qps_spike = df["IsSpikeByQps"].sum()
# print("Spike detection results - oryginalne:")
# print(f"Spike'ów wykrytych przez Latency: {latency_spike}")
# print(f"Spike'ów wykrytych przez CPU: {cpu_spike}")
# print(f"Spike'ów wykrytych przez QPS: {qps_spike}")
# print("Liczba spike'ów: ", latency_spike + cpu_spike + qps_spike)
# wykrywanie spike'ów
is_spike(df)
# is_spike(df,latency_spike,cpu_spike,qps_spike)
# 
# events_count = len(df)
# transactions_count = df["TransactionId"].nunique()
# correlations_count = df["CorrelationId"].nunique()

# print(f"Liczba eventów: {events_count}")
# print(f"Liczba transakcji (TransactionId): {transactions_count}")
# print(f"Liczba korelacji (CorrelationId): {correlations_count}")

# podstawowe informacje
# basic_info(df)


# binning wariancja entropia balance ratio
# b_v_e(df)

