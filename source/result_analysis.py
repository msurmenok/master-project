import pandas as pd

from ExperimentConfigs import configs

def compute_times_df(ldf):
    ldf["time_latency"] = ldf["time_reception"] - ldf["time_emit"]
    ldf["time_wait"] = ldf["time_in"] - ldf["time_reception"]
    ldf["time_service"] = ldf["time_out"] - ldf["time_in"]
    ldf["time_response"] = ldf["time_out"] - ldf["time_reception"]
    ldf["time_total_response"] = ldf["time_response"] + ldf["time_latency"]



for config in configs:
    print("----- Simulation Results for %s scenario" % config['scenario'])

    ff_avg_total_response = 0
    ff_avg_latency = 0

    memetic_avg_total_response = 0
    memetic_avg_latency = 0

    num_of_experiments = 10


for i in range(10):
    print("--------- %d Simulation Results ----------" % i)
    df_firstfit = pd.read_csv("results/Results_FirstFit_small_10000_%d.csv" % i)

    df_memetic = pd.read_csv("results/Results_Memetic_small_10000_%d.csv" % i)


    compute_times_df(df_firstfit)
    print("FF total time total response", df_firstfit["time_total_response"].sum())

    ff_avg_total_response += df_firstfit["time_total_response"].sum()
    ff_avg_latency += df_firstfit["time_latency"].mean()

    compute_times_df(df_memetic)
    print("Memetic total time total response", df_memetic["time_total_response"].sum())

    memetic_avg_latency += df_memetic["time_latency"].mean()
    memetic_avg_total_response += df_memetic["time_total_response"].sum()

print("=============== %d experiments average =====================" % num_of_experiments)
print("first fit total time response (avg of %d experiments) = %d " % (num_of_experiments, ff_avg_total_response / num_of_experiments))
# print("first fit avg latency = ", ff_avg_latency / 3)

print("memetic total time response (avg of %d experiments) = %d" % (num_of_experiments, memetic_avg_total_response / num_of_experiments))
# print("memetic avg latency = ", memetic_avg_latency / 3)

