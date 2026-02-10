import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from basic_info.index import basic_info
from binning_var_entropy.index import b_v_e
from anomaly_detection.index import anomaly_detection

df = pd.read_csv("logs.csv",sep=";")

df["AttributesJson"] = df["AttributesJson"].apply(json.loads)

attributes_df = pd.json_normalize(df["AttributesJson"])

df = pd.concat(
    [df.drop(columns=["AttributesJson"]), attributes_df],
    axis=1
)

# podstawowe informacje
# basic_info(df)


#binning wariancja entropia balance ratio
# b_v_e(df)

anomaly_detection(df)