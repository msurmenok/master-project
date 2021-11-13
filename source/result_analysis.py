import os

from functools import partial
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json

from ExperimentConfigs import configs


def filter_by_deadline(row, a):
    if row["time_total_response"] <= a[row["app"]]["deadline"]:
        return True
    else:
        return False


def compute_times_df(ldf):
    ldf["time_latency"] = ldf["time_reception"] - ldf["time_emit"]
    ldf["time_wait"] = ldf["time_in"] - ldf["time_reception"]
    ldf["time_service"] = ldf["time_out"] - ldf["time_in"]
    ldf["time_response"] = ldf["time_out"] - ldf["time_reception"]
    ldf["time_total_response"] = ldf["time_response"] + ldf["time_latency"]


# MemeticExperimental == MemeticExperimental1 in 'algorithm_time.csv'
algorithms = ['FirstFitRAM', 'FirstFitTime', 'MemeticWithoutLocalSearch', 'MemeticExperimental',
              'MemeticExperimental2', 'MemeticExperimental3', 'MemeticExperimental4', 'MemeticExperimental5',
              'MemeticExperimental6', 'MemeticExperimental7', 'MemeticExperimental8', 'Memetic']

# TODO change back to configs when large experiment are done
# configs2 = [
#     {
#         'scenario': 'tiny',
#         'iterations': 24
#     },
#     {
#         'scenario': 'small',
#         'iterations': 24
#     },
#     {
#         'scenario': 'medium',
#         'iterations': 24
#     },
    # {
    #     'scenario': 'large',
    #     'iterations': 16
    # },
    # {
    #     'scenario': 'verylarge',
    #     'iterations': 5
    # }
# ]

for config in configs:

    print("----- Simulation Results for %s scenario" % config['scenario'])
    # simulation_time = 10000
    simulation_time = 100000
    num_of_experiments = config['iterations']
    results_folder = "results/current/"

    # open files to store the result and write headers
    results_analysis_file = results_folder + "analysis.csv"
    results_analysis_average_file = results_folder + "analysis_averages.csv"

    file = open(results_analysis_file, 'a+')  # save completion time
    file.write(
        'scenario,# experiment,algorithm,mean latency,min latency,max latency,average total response,num requests, num failed requests\n')
    file_averages = open(results_analysis_average_file, 'a+')
    file_averages.write(
        'scenario,# experiments,algorithm,average average latency, median average latency, 75 perc average latency, average average total response, median average total response, 75 perc average total response,avg request failed to total ratio,median request failed to total ratio,avg number succ requests,median number succ requests,average calculation time,average num services on fog\n')

    for algorithm in algorithms:
        avg_latency = []
        avg_totalresponse = []
        num_requests = []
        num_failed_requests = []
        calculation_time = []
        services_on_fog = []

        for i in range(num_of_experiments):
            # calculate total time taken by each algorithm:
            algorithm_time_filepath = results_folder + 'results_' + config['scenario'] + '_' + str(
                i) + '/algorithm_time.csv'

            data = pd.read_csv(algorithm_time_filepath, header=None)
            algorithm_to_time = list(zip(data[1].to_list(), data[2].to_list(), data[3].to_list()))

            algorithm_name_in_algorithm_time = algorithm
            if algorithm == 'MemeticExperimental':
                algorithm_name_in_algorithm_time = 'MemeticExperimental1'

            for tuple in algorithm_to_time:
                if tuple[0] == algorithm_name_in_algorithm_time:
                    calculation_time.append(tuple[1])
                    services_on_fog.append(tuple[2])
                    break

            # pull the information about application deadlines
            data_folder = "data/data_" + config['scenario'] + "_" + str(i)
            appDefinitionFile = open(data_folder + '/appDefinition.json')
            appDefinition = json.load(appDefinitionFile)

            # create partially applied function and provide information abot app deadlines for this experiment
            filter_fn = partial(filter_by_deadline, a=appDefinition)

            df = pd.read_csv(
                (results_folder + "/results_%s_%dResults_%s_%s_%d_%d.csv") % (
                    config['scenario'], i, algorithm, config['scenario'], simulation_time, i))
            compute_times_df(df)

            num_requests.append(len(df))

            # create mask to keep only requests that were completed within a deadline
            df_mask = df.apply(filter_fn, axis=1)

            df_failed_requests = df_mask[df_mask == False]
            num_failed_requests.append(len(df_failed_requests))

            df_within_deadline = df[df_mask]

            avg_latency.append(df_within_deadline["time_latency"].mean())
            avg_totalresponse.append(df_within_deadline["time_total_response"].mean())

            file.write('%s,experiment_%d,%s,%f,%f,%f,%f,%d,%d\n' % (
                config['scenario'], i, algorithm,
                df_within_deadline["time_latency"].mean(),
                df_within_deadline["time_latency"].min(),
                df_within_deadline["time_latency"].max(),
                df_within_deadline["time_total_response"].mean(),
                len(df),
                len(df_failed_requests)))

        plt.hist(avg_latency, bins=20)
        plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_latency_' + algorithm + '.png')
        plt.clf()
        plt.hist(avg_totalresponse, bins=20)
        plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_totalresponse_' + algorithm + '.png')
        plt.clf()

        avg_avg_latency = float(np.mean(avg_latency))
        p50_avg_latency = float(np.percentile(avg_latency, 50))
        p75_avg_latency = float(np.percentile(avg_latency, 75))

        avg_avg_totalresponse = float(np.mean(avg_totalresponse))
        p50_avg_totalresponse = float(np.percentile(avg_totalresponse, 50))
        p75_avg_totalresponse = float(np.percentile(avg_totalresponse, 75))

        request_failed_to_total_ratio = float(np.mean(np.array(num_failed_requests) / np.array(num_requests)))
        p50_request_failed_to_total_ratio = float(
            np.percentile(np.array(num_failed_requests) / np.array(num_requests), 50))

        avg_num_succ_requests = float(np.mean(np.array(num_requests) - np.array(num_failed_requests)))
        p50_num_succ_requests = float(np.percentile(np.array(num_requests) - np.array(num_failed_requests), 50))

        file_averages.write('%s,%d,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
            config['scenario'], num_of_experiments, algorithm,
            avg_avg_latency,
            p50_avg_latency,
            p75_avg_latency,
            avg_avg_totalresponse,
            p50_avg_totalresponse,
            p75_avg_totalresponse,
            request_failed_to_total_ratio,
            p50_request_failed_to_total_ratio,
            avg_num_succ_requests,
            p50_num_succ_requests,
            sum(calculation_time) / num_of_experiments, sum(services_on_fog) / num_of_experiments))
    file.close()
    file_averages.close()
