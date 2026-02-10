import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


df = pd.read_csv("logs.csv",sep=";")

df["AttributesJson"] = df["AttributesJson"].apply(json.loads)

attributes_df = pd.json_normalize(df["AttributesJson"])

df = pd.concat(
    [df.drop(columns=["AttributesJson"]), attributes_df],
    axis=1
)

events_count = len(df)
transactions_count = df["TransactionId"].nunique()
correlations_count = df["CorrelationId"].nunique()

print(f"Liczba eventów: {events_count}")
print(f"Liczba transakcji (TransactionId): {transactions_count}")
print(f"Liczba korelacji (CorrelationId): {correlations_count}")