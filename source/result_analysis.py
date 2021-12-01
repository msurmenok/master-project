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


def filter_by_importance(row, a, max_priority):
    priority = a[row['app']]['module'][0]['PRIORITY']
    if priority == max_priority:
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
algorithms = ['MemeticExperimental2', 'MemeticExperimental6', 'MemeticExperimental8', 'Memetic']
algorithms = ['FirstFitRAM', 'FirstFitTime', 'MemeticWithoutLocalSearch', 'MemeticExperimental',
                  'MemeticExperimental2', 'MemeticExperimental3', 'MemeticExperimental4', 'MemeticExperimental5',
                  'MemeticExperimental6', 'MemeticExperimental7', 'MemeticExperimental8',
                  'Memetic']


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
    # results_folder = "results_slow_traffic/current/"
    results_folder = "results_high_traffic/current/"
    # results_folder = "results_small_hosts/current/"
    # results_folder = "results_big_hosts/current/"
    results_folder = 'results_average/current/'

    # open files to store the result and write headers
    results_analysis_file = results_folder + "analysis2.csv"
    results_analysis_average_file = results_folder + "analysis_averages2.csv"

    file = open(results_analysis_file, 'a+')  # save completion time
    file.write(
        'scenario,# experiment,algorithm,mean latency,min latency,max latency,average total response,num requests, num failed requests\n')
    file_averages = open(results_analysis_average_file, 'a+')
    file_averages.write(
        'scenario,# experiments,algorithm,average average latency, median average latency,average average total response, median average total response,std total response,avg request failed to total ratio,std failed ratio,important services average average total response,std important services total response,avg important requests failed to total ratio,std important failed ratio,num hosts used,average calculation time,average num services on fog\n')

    for algorithm in algorithms:
        avg_latency = []
        avg_totalresponse = []
        important_totalresponse = []
        num_requests = []
        num_failed_requests = []
        num_important_requests = []
        num_important_failed_requests = []
        calculation_time = []
        services_on_fog = []
        num_unique_hosts = []

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
            # data_folder = "data_slow_traffic/data_" + config['scenario'] + "_" + str(i)
            data_folder = "data_high_traffic/data_" + config['scenario'] + "_" + str(i)
            # data_folder = "data_small_hosts/data_" + config['scenario'] + "_" + str(i)
            # data_folder = "data_big_hosts/data_" + config['scenario'] + "_" + str(i)
            # data_folder = 'data_average/data_' + config['scenario'] + '_' + str(i)
            appDefinitionFile = open(data_folder + '/appDefinition.json')
            appDefinition = json.load(appDefinitionFile)

            # create partially applied function and provide information about app deadlines for this experiment
            filter_fn = partial(filter_by_deadline, a=appDefinition)

            df = pd.read_csv(
                (results_folder + "/results_%s_%dResults_%s_%s_%d_%d.csv") % (
                    config['scenario'], i, algorithm, config['scenario'], simulation_time, i))
            compute_times_df(df)

            num_requests.append(len(df))

            # num of fog devices used
            # find what is cloud id
            netDefinitionFile = open(data_folder + '/netDefinition.json')
            netDefinition = json.load(netDefinitionFile)
            # find all unique hosts
            all_hosts = df["TOPO.dst"]
            unique_hosts = all_hosts.unique()
            total_num_unique_hosts = 0

            for host in unique_hosts:
                if netDefinition["entity"][host]['type'] == 'CLOUD':
                    print('host is Cloud')
                if netDefinition["entity"][host]['type'] != 'CLOUD':
                    total_num_unique_hosts += 1

            num_unique_hosts.append(total_num_unique_hosts)
            # create mask to keep only requests that were completed within a deadline
            df_mask = df.apply(filter_fn, axis=1)

            df_failed_requests = df_mask[df_mask == False]
            num_failed_requests.append(len(df_failed_requests))

            df_within_deadline = df[df_mask]

            avg_latency.append(df_within_deadline["time_latency"].mean())
            avg_totalresponse.append(df_within_deadline["time_total_response"].mean())

            # filter by request with the maximum priority
            max_priority = 1
            filter_fn_2 = partial(filter_by_importance, a=appDefinition, max_priority=max_priority)
            df_important_mask = df.apply(filter_fn_2, axis=1)
            df_important = df[df_important_mask]
            num_important_requests.append(len(df_important))

            df_important_failed_mask = df_important.apply(filter_fn, axis=1)
            df_important_failed = df_important_failed_mask[df_important_failed_mask == False]
            num_important_failed_requests.append(len(df_important_failed))

            important_totalresponse.append(df_important["time_total_response"].mean())

            file.write('%s,experiment_%d,%s,%f,%f,%f,%f,%d,%d\n' % (
                config['scenario'], i, algorithm,
                df_within_deadline["time_latency"].mean(),
                df_within_deadline["time_latency"].min(),
                df_within_deadline["time_latency"].max(),
                df_within_deadline["time_total_response"].mean(),
                len(df),
                len(df_failed_requests)))

        plt.hist(avg_latency, bins=20)

        if not os.path.exists(results_folder + 'histograms/'):
            os.makedirs(results_folder + 'histograms/')

        plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_latency_' + algorithm + '.png')
        plt.clf()
        plt.hist(avg_totalresponse, bins=20)
        plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_totalresponse_' + algorithm + '.png')
        plt.clf()

        # latency
        avg_avg_latency = float(np.mean(avg_latency))
        p50_avg_latency = float(np.percentile(avg_latency, 50))

        # total response
        avg_avg_totalresponse = float(np.mean(avg_totalresponse))
        std_totalresponse = float(np.std(avg_totalresponse))
        p50_avg_totalresponse = float(np.percentile(avg_totalresponse, 50))

        # total response for important services
        avg_avg_important_totalresponse = float(np.mean(important_totalresponse))
        std_important_totalresponse = float(np.std(important_totalresponse))

        important_request_failed_to_total_ratio = float(
            np.mean(np.array(num_important_failed_requests) / np.array(num_important_requests)))
        std_important_failed_to_total_ratio = float(
            np.std(np.mean(np.array(num_important_failed_requests) / np.array(num_important_requests))))

        # failed requests ratio (both important and not important)
        request_failed_to_total_ratio = float(np.mean(np.array(num_failed_requests) / np.array(num_requests)))
        std_failed_to_total_ratio = float(np.std(np.mean(np.array(num_failed_requests) / np.array(num_requests))))
        average_num_hosts = float(np.mean(num_unique_hosts))

        # scenario,# experiments,algorithm,average average latency, median average latency,average average total response,
        # median average total response,std total response,
        # avg request failed to total ratio,std failed ratio,
        # important services average average total response,std important services total response,

        # avg important requests failed to total ratio,std important failed ratio,
        #
        # num hosts used,average calculation time,average num services on fog\n')
        file_averages.write('%s,%d,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
            config['scenario'], num_of_experiments, algorithm,
            avg_avg_latency,
            p50_avg_latency,
            avg_avg_totalresponse,
            p50_avg_totalresponse,
            std_totalresponse,
            request_failed_to_total_ratio * 100,
            std_failed_to_total_ratio * 100,
            avg_avg_important_totalresponse,
            std_important_totalresponse,
            important_request_failed_to_total_ratio * 100,
            std_important_failed_to_total_ratio * 100,
            average_num_hosts,
            sum(calculation_time) / num_of_experiments, sum(services_on_fog) / num_of_experiments))
    file.close()
    file_averages.close()
