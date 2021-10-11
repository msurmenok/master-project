import os

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
    simulation_time = 10000
    # simulation_time = 100000

    # to calculate average of time latency over all experiments for this scenario
    ff_ram_all_experiments_avg_latency = 0
    ff_time_all_experiments_avg_latency = 0
    memetic_no_local_search_all_experiments_avg_latency = 0
    memetic_all_experiments_avg_latency = 0

    num_of_experiments = config['iterations']

    results_folder = "results"
    for i in range(num_of_experiments):
        # First Fit by RAM
        print("--------- %d Simulation Results ----------" % i)

        df_firstfit_ram = pd.read_csv(
            (results_folder + "/Results_FirstFitRAM_%s_%d_%d.csv") % (config['scenario'], simulation_time, i))
        compute_times_df(df_firstfit_ram)

        ff_ram_all_experiments_avg_latency += df_firstfit_ram["time_latency"].mean()

        print("FF RAM number of tasks processed: %d" % len(df_firstfit_ram))
        print("FF RAM average latency", df_firstfit_ram["time_latency"].mean())
        print("FF RAM minimum latency", df_firstfit_ram["time_latency"].min())
        print("FF RAM maximum latency", df_firstfit_ram["time_latency"].max())

        print("FF RAM total time total response", df_firstfit_ram["time_total_response"].sum())
        print("FF RAM mean time total response", df_firstfit_ram["time_total_response"].mean())
        print()

        # First Fit by Time
        df_firstfit_time = pd.read_csv(
            (results_folder + "/Results_FirstFitTime_%s_%d_%d.csv") % (config['scenario'], simulation_time, i))
        compute_times_df(df_firstfit_time)
        ff_time_all_experiments_avg_latency += df_firstfit_time["time_latency"].mean()

        print("FF Time number of tasks processed: %d" % len(df_firstfit_time))
        print("FF Time average latency", df_firstfit_time["time_latency"].mean())
        print("FF Time minimum latency", df_firstfit_time["time_latency"].min())
        print("FF Time maximum latency", df_firstfit_time["time_latency"].max())

        print("FF Time total time total response", df_firstfit_time["time_total_response"].sum())
        print("FF Time mean time total response", df_firstfit_time["time_total_response"].mean())
        print()

        # Memetic without Local Search
        df_memetic_no_local_search = pd.read_csv((results_folder + "/Results_MemeticWithoutLocalSearch_%s_%d_%d.csv") % (config['scenario'], simulation_time, i))
        compute_times_df(df_memetic_no_local_search)

        memetic_no_local_search_all_experiments_avg_latency += df_memetic_no_local_search["time_latency"].mean()
        print("Memetic w/o LC number of tasks processed: %d" % len(df_memetic_no_local_search))
        print("Memetic w/o LC average latency", df_memetic_no_local_search["time_latency"].mean())
        print("Memetic w/o LC minimum latency", df_memetic_no_local_search["time_latency"].min())
        print("Memetic w/o LC maximum latency", df_memetic_no_local_search["time_latency"].max())

        print("Memetic w/o LC total time total response", df_memetic_no_local_search["time_total_response"].sum())
        print("Memetic w/o LC mean time total response", df_memetic_no_local_search["time_total_response"].mean())
        print()

        # Memetic (with local search)
        df_memetic = pd.read_csv((results_folder + "/Results_Memetic_%s_%d_%d.csv") % (config['scenario'], simulation_time, i))
        compute_times_df(df_memetic)

        memetic_all_experiments_avg_latency += df_memetic["time_latency"].mean()
        print("Memetic number of tasks processed: %d" % len(df_memetic))
        print("Memetic average latency", df_memetic["time_latency"].mean())
        print("Memetic minimum latency", df_memetic["time_latency"].min())
        print("Memetic maximum latency", df_memetic["time_latency"].max())

        print("Memetic total time total response", df_memetic["time_total_response"].sum())
        print("Memetic mean time total response", df_memetic["time_total_response"].mean())
        print()

    print("=============== %d experiments average =====================" % num_of_experiments)
    print("first fit ram total time response (avg of %d experiments) = %d " % (
        num_of_experiments, ff_ram_all_experiments_avg_latency / num_of_experiments))

    print("first fit time total time response (avg of %d experiments) = %d " % (
        num_of_experiments, ff_ram_all_experiments_avg_latency / num_of_experiments))

    print("memetic total time response (avg of %d experiments) = %d" % (
        num_of_experiments, memetic_all_experiments_avg_latency / num_of_experiments))
    print("\n\n")
