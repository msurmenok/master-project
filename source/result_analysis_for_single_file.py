import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from ExperimentConfigs import configs


def compute_times_df(ldf):
    ldf["time_latency"] = ldf["time_reception"] - ldf["time_emit"]
    ldf["time_wait"] = ldf["time_in"] - ldf["time_reception"]
    ldf["time_service"] = ldf["time_out"] - ldf["time_in"]
    ldf["time_response"] = ldf["time_out"] - ldf["time_reception"]
    ldf["time_total_response"] = ldf["time_response"] + ldf["time_latency"]


results_folder = "results/current/"
file_to_analyze = results_folder + 'results_tiny_0Results_Memetic_tiny_100000_0.csv'
df = pd.read_csv(file_to_analyze)
compute_times_df(df)
print(len(df[df.time_latency > 20000]))
print(len(df))
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
mydf = df[df.time_latency > 20000]
print(mydf[["module", "TOPO.src", "TOPO.dst", "time_emit", "time_reception"]])
# print(mydf.head())
