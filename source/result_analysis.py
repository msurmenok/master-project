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
              'MemeticExperimental6', 'MemeticExperimental7', 'MemeticExperimental8', 'MemeticExperimental9', 'Memetic']

# TODO change back to configs when large experiment are done
configs2 = [
    {
        'scenario': 'tiny',
        'iterations': 24
    },
    {
        'scenario': 'small',
        'iterations': 24
    },
    {
        'scenario': 'medium',
        'iterations': 24
    },
    # {
    #     'scenario': 'large',
    #     'iterations': 16
    # },
    # {
    #     'scenario': 'verylarge',
    #     'iterations': 5
    # }
]

for config in configs2:

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

    # to calculate average of time latency over all experiments for this scenario
    # ff_ram_all_experiments_avg_latency = []
    # ff_time_all_experiments_avg_latency = []
    # memetic_no_local_search_all_experiments_avg_latency = []
    # memetic_experemental_all_experiments_avg_latency = []
    # memetic_experemental2_all_experiments_avg_latency = []
    # memetic_experemental3_all_experiments_avg_latency = []
    # memetic_experemental4_all_experiments_avg_latency = []
    # memetic_experemental5_all_experiments_avg_latency = []
    # memetic_experemental6_all_experiments_avg_latency = []
    # memetic_experemental7_all_experiments_avg_latency = []
    # memetic_experemental8_all_experiments_avg_latency = []
    # memetic_experemental9_all_experiments_avg_latency = []
    # memetic_all_experiments_avg_latency = []
    #
    # ff_ram_all_experiments_avg_totalresponse = []
    # ff_time_all_experiments_avg_totalresponse = []
    # memetic_no_local_search_all_experiments_avg_totalresponse = []
    # memetic_experemental_all_experiments_avg_totalresponse = []
    # memetic_experemental2_all_experiments_avg_totalresponse = []
    # memetic_experemental3_all_experiments_avg_totalresponse = []
    # memetic_experemental4_all_experiments_avg_totalresponse = []
    # memetic_experemental5_all_experiments_avg_totalresponse = []
    # memetic_experemental6_all_experiments_avg_totalresponse = []
    # memetic_experemental7_all_experiments_avg_totalresponse = []
    # memetic_experemental8_all_experiments_avg_totalresponse = []
    # memetic_experemental9_all_experiments_avg_totalresponse = []
    # memetic_all_experiments_avg_totalresponse = []
    #
    # ff_ram_num_requests = 0
    # ff_time_num_requests = 0
    # memetic_no_local_search_num_requests = 0
    # memetic_experemental_num_requests = 0
    # memetic_experemental2_num_requests = 0
    # memetic_experemental3_num_requests = 0
    # memetic_experemental4_num_requests = 0
    # memetic_experemental5_num_requests = 0
    # memetic_experemental6_num_requests = 0
    # memetic_experemental7_num_requests = 0
    # memetic_experemental8_num_requests = 0
    # memetic_experemental9_num_requests = 0
    # memetic_num_requests = 0
    #
    # ff_ram_num_failed_requests = 0
    # ff_time_num_failed_requests = 0
    # memetic_no_local_search_num_failed_requests = 0
    # memetic_experemental_num_failed_requests = 0
    # memetic_experemental2_num_failed_requests = 0
    # memetic_experemental3_num_failed_requests = 0
    # memetic_experemental4_num_failed_requests = 0
    # memetic_experemental5_num_failed_requests = 0
    # memetic_experemental6_num_failed_requests = 0
    # memetic_experemental7_num_failed_requests = 0
    # memetic_experemental8_num_failed_requests = 0
    # memetic_experemental9_num_failed_requests = 0
    # memetic_num_failed_requests = 0
    #
    # ff_ram_calculation_time = 0
    # ff_time_calculation_time = 0
    # memetic_no_local_search_calculation_time = 0
    # memetic_experimental_calculation_time = 0
    # memetic_experimental2_calculation_time = 0
    # memetic_experimental3_calculation_time = 0
    # memetic_experimental4_calculation_time = 0
    # memetic_experimental5_calculation_time = 0
    # memetic_experimental6_calculation_time = 0
    # memetic_experimental7_calculation_time = 0
    # memetic_experimental8_calculation_time = 0
    # memetic_experimental9_calculation_time = 0
    # memetic_calculation_time = 0
    #
    # ff_ram_services_on_fog = 0
    # ff_time_services_on_fog = 0
    # memetic_no_local_search_services_on_fog = 0
    # memetic_experimental_services_on_fog = 0
    # memetic_experimental2_services_on_fog = 0
    # memetic_experimental3_services_on_fog = 0
    # memetic_experimental4_services_on_fog = 0
    # memetic_experimental5_services_on_fog = 0
    # memetic_experimental6_services_on_fog = 0
    # memetic_experimental7_services_on_fog = 0
    # memetic_experimental8_services_on_fog = 0
    # memetic_experimental9_services_on_fog = 0
    # memetic_services_on_fog = 0
    #
    # num_of_experiments = config['iterations']
    # results_folder = "results/current/"

    # results_analysis_file = results_folder + "analysis.csv"
    # results_analysis_average_file = results_folder + "analysis_averages.csv"
    #
    # file = open(results_analysis_file, 'a+')  # save completion time
    # file.write(
    #     'scenario,# experiment,algorithm,mean latency,min latency,max latency,average total response,num requests\n')
    # file_averages = open(results_analysis_average_file, 'a+')
    # file_averages.write(
    #     'scenario,# experiments,algorithm,average average latency, median average latency, 75 perc average latency, average average total response, median average total response, 75 perc average total response, average num requests,average num of failed requests, average calculation time,average num services on fog\n')

    # for i in range(num_of_experiments):
    # calculate total time taken by each algorithm:
    # algorithm_time_filepath = results_folder + 'results_' + config['scenario'] + '_' + str(
    #     i) + '/algorithm_time.csv'
    #
    # data = pd.read_csv(algorithm_time_filepath, header=None)
    # algorithm_to_time = list(zip(data[1].to_list(), data[2].to_list(), data[3].to_list()))
    # for tuple in algorithm_to_time:
    #     if tuple[0] == 'FirstFitRAM':
    #         ff_ram_calculation_time += tuple[1]
    #         ff_ram_services_on_fog += tuple[2]
    #     elif tuple[0] == 'FirstFitTime':
    #         ff_time_calculation_time += tuple[1]
    #         ff_time_services_on_fog += tuple[2]
    #     elif tuple[0] == 'MemeticWithoutLocalSearch':
    #         memetic_no_local_search_calculation_time += tuple[1]
    #         memetic_no_local_search_services_on_fog += tuple[2]
    #     elif tuple[0] == 'MemeticExperimental1':
    #         memetic_experimental_calculation_time += tuple[1]
    #         memetic_experimental_services_on_fog += tuple[2]
    #     elif tuple[0] == 'MemeticExperimental2':
    #         memetic_experimental2_calculation_time += tuple[1]
    #         memetic_experimental2_services_on_fog += tuple[2]
    #     elif tuple[0] == 'MemeticExperimental3':
    #         memetic_experimental3_calculation_time += tuple[1]
    #         memetic_experimental3_services_on_fog += tuple[2]
    #     elif tuple[0] == 'MemeticExperimental4':
    #         memetic_experimental4_calculation_time += tuple[1]
    #         memetic_experimental4_services_on_fog += tuple[2]
    #     elif tuple[0] == 'MemeticExperimental5':
    #         memetic_experimental5_calculation_time += tuple[1]
    #         memetic_experimental5_services_on_fog += tuple[2]
    #     elif tuple[0] == 'MemeticExperimental6':
    #         memetic_experimental6_calculation_time += tuple[1]
    #         memetic_experimental6_services_on_fog += tuple[2]
    #     elif tuple[0] == 'Memetic':
    #         memetic_calculation_time += tuple[1]
    #         memetic_services_on_fog += tuple[2]

    # load service ids and deadlines
    # data_folder = "data/data_" + config['scenario'] + "_" + str(i)
    # appDefinitionFile = open(data_folder + '/appDefinition.json')
    # appDefinition = json.load(appDefinitionFile)
    # print(appDefinition[0]['deadline'])
    #
    # filter_fn = partial(filter_by_deadline, a=appDefinition)
    # # df = df[df.apply(filter_fn, axis=1)]

    # First Fit by RAM
    # print("--------- %d Simulation Results ----------" % i)

    # df_firstfit_ram = pd.read_csv(
    #     (results_folder + "/results_%s_%dResults_FirstFitRAM_%s_%d_%d.csv") % (
    #         config['scenario'], i, config['scenario'], simulation_time, i))
    # compute_times_df(df_firstfit_ram)

    # num_requests = len(df_firstfit_ram)
    # ff_ram_num_requests += num_requests

    # df_firstfit_ram_mask = df_firstfit_ram.apply(filter_fn, axis=1)
    #
    # failed_requests = df_firstfit_ram_mask[df_firstfit_ram_mask == False]
    # ff_ram_num_failed_requests += len(failed_requests)
    #
    # # filter all requests failed by deadline
    # df_firstfit_ram = df_firstfit_ram[df_firstfit_ram_mask]
    # ff_ram_all_experiments_avg_latency.append(df_firstfit_ram["time_latency"].mean())
    # ff_ram_all_experiments_avg_totalresponse.append(df_firstfit_ram["time_total_response"].mean())

    # print("FF RAM number of tasks processed: %d" % len(df_firstfit_ram))
    # print("FF RAM average latency", df_firstfit_ram["time_latency"].mean())
    # print("FF RAM minimum latency", df_firstfit_ram["time_latency"].min())
    # print("FF RAM maximum latency", df_firstfit_ram["time_latency"].max())
    #
    # print("FF RAM total time total response", df_firstfit_ram["time_total_response"].sum())
    # print("FF RAM mean time total response", df_firstfit_ram["time_total_response"].mean())
    # print()

    # file.write('%s,experiment_%d,FF RAM,%f,%f,%f,%f,%d\n' % (
    #     config['scenario'], i, df_firstfit_ram["time_latency"].mean(), df_firstfit_ram["time_latency"].min(),
    #     df_firstfit_ram["time_latency"].max(), df_firstfit_ram["time_total_response"].mean(), len(df_firstfit_ram)))

    # First Fit by Time
    # df_firstfit_time = pd.read_csv(
    #     (results_folder + "/results_%s_%dResults_FirstFitTime_%s_%d_%d.csv") % (
    #         config['scenario'], i, config['scenario'], simulation_time, i))
    # compute_times_df(df_firstfit_time)
    # ff_time_all_experiments_avg_latency.append(df_firstfit_time["time_latency"].mean())
    # ff_time_all_experiments_avg_totalresponse.append(df_firstfit_time["time_total_response"].mean())
    # ff_time_num_requests += len(df_firstfit_time)
    #
    # print("FF Time number of tasks processed: %d" % len(df_firstfit_time))
    # print("FF Time average latency", df_firstfit_time["time_latency"].mean())
    # print("FF Time minimum latency", df_firstfit_time["time_latency"].min())
    # print("FF Time maximum latency", df_firstfit_time["time_latency"].max())
    #
    # print("FF Time total time total response", df_firstfit_time["time_total_response"].sum())
    # print("FF Time mean time total response", df_firstfit_time["time_total_response"].mean())
    # print()
    #
    # file.write('%s,experiment_%d,FF Time,%f,%f,%f,%f,%d\n' % (
    #     config['scenario'], i, df_firstfit_time["time_latency"].mean(), df_firstfit_time["time_latency"].min(),
    #     df_firstfit_time["time_latency"].max(), df_firstfit_time["time_total_response"].mean(),
    #     len(df_firstfit_time)))
    #
    # # Memetic without Local Search
    # df_memetic_no_local_search = pd.read_csv(
    #     (results_folder + "/results_%s_%dResults_MemeticWithoutLocalSearch_%s_%d_%d.csv") % (
    #         config['scenario'], i, config['scenario'], simulation_time, i))
    # compute_times_df(df_memetic_no_local_search)
    #
    # memetic_no_local_search_all_experiments_avg_latency.append(df_memetic_no_local_search["time_latency"].mean())
    # memetic_no_local_search_all_experiments_avg_totalresponse.append(df_memetic_no_local_search[
    #                                                                      "time_total_response"].mean())
    # memetic_no_local_search_num_requests += len(df_memetic_no_local_search)
    #
    # print("Memetic w/o LC number of tasks processed: %d" % len(df_memetic_no_local_search))
    # print("Memetic w/o LC average latency", df_memetic_no_local_search["time_latency"].mean())
    # print("Memetic w/o LC minimum latency", df_memetic_no_local_search["time_latency"].min())
    # print("Memetic w/o LC maximum latency", df_memetic_no_local_search["time_latency"].max())
    #
    # print("Memetic w/o LC total time total response", df_memetic_no_local_search["time_total_response"].sum())
    # print("Memetic w/o LC mean time total response", df_memetic_no_local_search["time_total_response"].mean())
    # print()
    #
    # file.write('%s,experiment_%d,Memetic w/o LC,%f,%f,%f,%f,%d\n' % (
    #     config['scenario'], i, df_memetic_no_local_search["time_latency"].mean(),
    #     df_memetic_no_local_search["time_latency"].min(),
    #     df_memetic_no_local_search["time_latency"].max(), df_memetic_no_local_search["time_total_response"].mean(),
    #     len(df_memetic_no_local_search)))
    #
    # # Memetic Experimental
    # df_memetic_experimental = pd.read_csv(
    #     (results_folder + "/results_%s_%dResults_MemeticExperimental_%s_%d_%d.csv") % (
    #         config['scenario'], i, config['scenario'], simulation_time, i))
    # compute_times_df(df_memetic_experimental)
    #
    # memetic_experemental_all_experiments_avg_latency.append(df_memetic_experimental["time_latency"].mean())
    # memetic_experemental_all_experiments_avg_totalresponse.append(
    #     df_memetic_experimental["time_total_response"].mean())
    # memetic_experemental_num_requests += len(df_memetic_experimental)
    #
    # print("Memetic Experimental number of tasks processed: %d" % len(df_memetic_experimental))
    # print("Memetic Experimental average latency", df_memetic_experimental["time_latency"].mean())
    # print("Memetic Experimental minimum latency", df_memetic_experimental["time_latency"].min())
    # print("Memetic Experimental maximum latency", df_memetic_experimental["time_latency"].max())
    #
    # print("Memetic Experimental total time total response", df_memetic_experimental["time_total_response"].sum())
    # print("Memetic Experimental mean time total response", df_memetic_experimental["time_total_response"].mean())
    # print()
    #
    # file.write('%s,experiment_%d,Memetic Experimental,%f,%f,%f,%f,%d\n' % (
    #     config['scenario'], i, df_memetic_experimental["time_latency"].mean(),
    #     df_memetic_experimental["time_latency"].min(),
    #     df_memetic_experimental["time_latency"].max(), df_memetic_experimental["time_total_response"].mean(),
    #     len(df_memetic_experimental)))
    #
    # # Memetic Experimental 2
    # df_memetic_experimental2 = pd.read_csv(
    #     (results_folder + "/results_%s_%dResults_MemeticExperimental2_%s_%d_%d.csv") % (
    #         config['scenario'], i, config['scenario'], simulation_time, i))
    # compute_times_df(df_memetic_experimental2)
    #
    # memetic_experemental2_all_experiments_avg_latency.append(df_memetic_experimental2["time_latency"].mean())
    # memetic_experemental2_all_experiments_avg_totalresponse.append(df_memetic_experimental2[
    #                                                                    "time_total_response"].mean())
    # memetic_experemental2_num_requests += len(df_memetic_experimental2)
    #
    # print("Memetic Experimental 2 number of tasks processed: %d" % len(df_memetic_experimental2))
    # print("Memetic Experimental 2 average latency", df_memetic_experimental2["time_latency"].mean())
    # print("Memetic Experimental 2 minimum latency", df_memetic_experimental2["time_latency"].min())
    # print("Memetic Experimental 2 maximum latency", df_memetic_experimental2["time_latency"].max())
    #
    # print("Memetic Experimental 2 total time total response", df_memetic_experimental2["time_total_response"].sum())
    # print("Memetic Experimental 2 mean time total response", df_memetic_experimental2["time_total_response"].mean())
    # print()
    #
    # file.write('%s,experiment_%d,Memetic Experimental 2,%f,%f,%f,%f,%d\n' % (
    #     config['scenario'], i, df_memetic_experimental2["time_latency"].mean(),
    #     df_memetic_experimental2["time_latency"].min(),
    #     df_memetic_experimental2["time_latency"].max(), df_memetic_experimental2["time_total_response"].mean(),
    #     len(df_memetic_experimental2)))
    #
    # # Memetic Experimental 3
    # df_memetic_experimental3 = pd.read_csv(
    #     (results_folder + "/results_%s_%dResults_MemeticExperimental3_%s_%d_%d.csv") % (
    #         config['scenario'], i, config['scenario'], simulation_time, i))
    # compute_times_df(df_memetic_experimental3)
    #
    # memetic_experemental3_all_experiments_avg_latency.append(df_memetic_experimental3["time_latency"].mean())
    # memetic_experemental3_all_experiments_avg_totalresponse.append(df_memetic_experimental3[
    #                                                                    "time_total_response"].mean())
    # memetic_experemental3_num_requests += len(df_memetic_experimental3)
    #
    # print("Memetic Experimental 3 number of tasks processed: %d" % len(df_memetic_experimental3))
    # print("Memetic Experimental 3 average latency", df_memetic_experimental3["time_latency"].mean())
    # print("Memetic Experimental 3 minimum latency", df_memetic_experimental3["time_latency"].min())
    # print("Memetic Experimental 3 maximum latency", df_memetic_experimental3["time_latency"].max())
    #
    # print("Memetic Experimental 3 total time total response", df_memetic_experimental3["time_total_response"].sum())
    # print("Memetic Experimental 3 mean time total response", df_memetic_experimental3["time_total_response"].mean())
    # print()
    #
    # file.write('%s,experiment_%d,Memetic Experimental 3,%f,%f,%f,%f,%d\n' % (
    #     config['scenario'], i, df_memetic_experimental3["time_latency"].mean(),
    #     df_memetic_experimental3["time_latency"].min(),
    #     df_memetic_experimental3["time_latency"].max(), df_memetic_experimental3["time_total_response"].mean(),
    #     len(df_memetic_experimental3)))
    #
    # # Memetic Experimental 4
    # df_memetic_experimental4 = pd.read_csv(
    #     (results_folder + "/results_%s_%dResults_MemeticExperimental4_%s_%d_%d.csv") % (
    #         config['scenario'], i, config['scenario'], simulation_time, i))
    # compute_times_df(df_memetic_experimental4)
    #
    # memetic_experemental4_all_experiments_avg_latency.append(df_memetic_experimental4["time_latency"].mean())
    # memetic_experemental4_all_experiments_avg_totalresponse.append(df_memetic_experimental4[
    #                                                                    "time_total_response"].mean())
    # memetic_experemental4_num_requests += len(df_memetic_experimental4)
    #
    # print("Memetic Experimental 4 number of tasks processed: %d" % len(df_memetic_experimental4))
    # print("Memetic Experimental 4 average latency", df_memetic_experimental4["time_latency"].mean())
    # print("Memetic Experimental 4 minimum latency", df_memetic_experimental4["time_latency"].min())
    # print("Memetic Experimental 4 maximum latency", df_memetic_experimental4["time_latency"].max())
    #
    # print("Memetic Experimental 4 total time total response", df_memetic_experimental4["time_total_response"].sum())
    # print("Memetic Experimental 4 mean time total response", df_memetic_experimental4["time_total_response"].mean())
    # print()
    #
    # file.write('%s,experiment_%d,Memetic Experimental 4,%f,%f,%f,%f,%d\n' % (
    #     config['scenario'], i, df_memetic_experimental4["time_latency"].mean(),
    #     df_memetic_experimental4["time_latency"].min(),
    #     df_memetic_experimental4["time_latency"].max(), df_memetic_experimental4["time_total_response"].mean(),
    #     len(df_memetic_experimental4)))
    #
    # # Memetic Experimental 5
    # df_memetic_experimental5 = pd.read_csv(
    #     (results_folder + "/results_%s_%dResults_MemeticExperimental5_%s_%d_%d.csv") % (
    #         config['scenario'], i, config['scenario'], simulation_time, i))
    # compute_times_df(df_memetic_experimental5)
    #
    # memetic_experemental5_all_experiments_avg_latency.append(df_memetic_experimental5["time_latency"].mean())
    # memetic_experemental5_all_experiments_avg_totalresponse.append(df_memetic_experimental5[
    #                                                                    "time_total_response"].mean())
    # memetic_experemental5_num_requests += len(df_memetic_experimental5)
    #
    # print("Memetic Experimental 5 number of tasks processed: %d" % len(df_memetic_experimental5))
    # print("Memetic Experimental 5 average latency", df_memetic_experimental5["time_latency"].mean())
    # print("Memetic Experimental 5 minimum latency", df_memetic_experimental5["time_latency"].min())
    # print("Memetic Experimental 5 maximum latency", df_memetic_experimental5["time_latency"].max())
    #
    # print("Memetic Experimental 5 total time total response", df_memetic_experimental5["time_total_response"].sum())
    # print("Memetic Experimental 5 mean time total response", df_memetic_experimental5["time_total_response"].mean())
    # print()
    #
    # file.write('%s,experiment_%d,Memetic Experimental 5,%f,%f,%f,%f,%d\n' % (
    #     config['scenario'], i, df_memetic_experimental5["time_latency"].mean(),
    #     df_memetic_experimental5["time_latency"].min(),
    #     df_memetic_experimental5["time_latency"].max(), df_memetic_experimental5["time_total_response"].mean(),
    #     len(df_memetic_experimental5)))
    #
    # # Memetic Experimental 6
    # df_memetic_experimental6 = pd.read_csv(
    #     (results_folder + "/results_%s_%dResults_MemeticExperimental6_%s_%d_%d.csv") % (
    #         config['scenario'], i, config['scenario'], simulation_time, i))
    # compute_times_df(df_memetic_experimental6)
    #
    # memetic_experemental6_all_experiments_avg_latency.append(df_memetic_experimental6["time_latency"].mean())
    # memetic_experemental6_all_experiments_avg_totalresponse.append(df_memetic_experimental6[
    #                                                                    "time_total_response"].mean())
    # memetic_experemental6_num_requests += len(df_memetic_experimental6)
    #
    # print("Memetic Experimental 6 number of tasks processed: %d" % len(df_memetic_experimental6))
    # print("Memetic Experimental 6 average latency", df_memetic_experimental6["time_latency"].mean())
    # print("Memetic Experimental 6 minimum latency", df_memetic_experimental6["time_latency"].min())
    # print("Memetic Experimental 6 maximum latency", df_memetic_experimental6["time_latency"].max())
    #
    # print("Memetic Experimental 6 total time total response", df_memetic_experimental6["time_total_response"].sum())
    # print("Memetic Experimental 6 mean time total response", df_memetic_experimental6["time_total_response"].mean())
    # print()
    #
    # file.write('%s,experiment_%d,Memetic Experimental 6,%f,%f,%f,%f,%d\n' % (
    #     config['scenario'], i, df_memetic_experimental6["time_latency"].mean(),
    #     df_memetic_experimental6["time_latency"].min(),
    #     df_memetic_experimental6["time_latency"].max(), df_memetic_experimental6["time_total_response"].mean(),
    #     len(df_memetic_experimental6)))
    #
    # # Memetic (with local search)
    # df_memetic = pd.read_csv(
    #     (results_folder + "/results_%s_%dResults_Memetic_%s_%d_%d.csv") % (
    #         config['scenario'], i, config['scenario'], simulation_time, i))
    # compute_times_df(df_memetic)
    #
    # memetic_all_experiments_avg_latency.append(df_memetic["time_latency"].mean())
    # memetic_all_experiments_avg_totalresponse.append(df_memetic["time_total_response"].mean())
    # memetic_num_requests += len(df_memetic)
    #
    # print("Memetic number of tasks processed: %d" % len(df_memetic))
    # print("Memetic average latency", df_memetic["time_latency"].mean())
    # print("Memetic minimum latency", df_memetic["time_latency"].min())
    # print("Memetic maximum latency", df_memetic["time_latency"].max())
    #
    # print("Memetic total time total response", df_memetic["time_total_response"].sum())
    # print("Memetic mean time total response", df_memetic["time_total_response"].mean())
    # print()
    # file.write('%s,experiment_%d,Memetic,%f,%f,%f,%f,%d\n' % (
    #     config['scenario'], i, df_memetic["time_latency"].mean(), df_memetic["time_latency"].min(),
    #     df_memetic["time_latency"].max(), df_memetic["time_total_response"].mean(), len(df_memetic)))

    # print("=============== %d experiments average =====================" % num_of_experiments)
    # plt.hist(ff_ram_all_experiments_avg_latency, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_latency_ff_ram.png')
    # plt.clf()
    #
    # plt.hist(ff_ram_all_experiments_avg_totalresponse, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_totalresponse_ff_ram.png')
    # plt.clf()

    # avg_avg_latency = float(np.mean(ff_ram_all_experiments_avg_latency))
    # p50_avg_latency = float(np.percentile(ff_ram_all_experiments_avg_latency, 50))
    # p75_avg_latency = float(np.percentile(ff_ram_all_experiments_avg_latency, 75))
    #
    # avg_avg_totalresponse = float(np.mean(ff_ram_all_experiments_avg_totalresponse))
    # p50_avg_totalresponse = float(np.percentile(ff_ram_all_experiments_avg_totalresponse, 50))
    # p75_avg_totalresponse = float(np.percentile(ff_ram_all_experiments_avg_totalresponse, 75))
    #
    # print("first fit ram avg latency (avg of %d experiments) = %f " % (
    #     num_of_experiments, avg_avg_latency))
    # print("first fit ram avg total response time (avg of %d experiments) = %f " % (
    #     num_of_experiments, avg_avg_totalresponse))
    # print("first fit ram avg num of requests (avg of %d experiments) = %f " % (
    #     num_of_experiments, ff_ram_num_requests / num_of_experiments))
    #
    # file_averages.write('%s,%d,FF RAM,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
    #     config['scenario'], num_of_experiments,
    #     avg_avg_latency,
    #     p50_avg_latency,
    #     p75_avg_latency,
    #     avg_avg_totalresponse,
    #     p50_avg_totalresponse,
    #     p75_avg_totalresponse,
    #     ff_ram_num_requests / num_of_experiments,
    #     ff_ram_calculation_time / num_of_experiments, ff_ram_services_on_fog / num_of_experiments))
    #
    # # first fit time averages
    # plt.hist(ff_time_all_experiments_avg_latency, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_latency_ff_time.png')
    # plt.clf()
    #
    # plt.hist(ff_time_all_experiments_avg_totalresponse, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_totalresponse_ff_time.png')
    # plt.clf()
    #
    # avg_avg_latency = float(np.mean(ff_time_all_experiments_avg_latency))
    # p50_avg_latency = float(np.percentile(ff_time_all_experiments_avg_latency, 50))
    # p75_avg_latency = float(np.percentile(ff_time_all_experiments_avg_latency, 75))

    # avg_avg_totalresponse = float(np.mean(ff_time_all_experiments_avg_totalresponse))
    # p50_avg_totalresponse = float(np.percentile(ff_time_all_experiments_avg_totalresponse, 50))
    # p75_avg_totalresponse = float(np.percentile(ff_time_all_experiments_avg_totalresponse, 75))
    #
    # print("first fit time avg latency (avg of %d experiments) = %f " % (
    #     num_of_experiments, avg_avg_latency))
    # print("first fit time avg total response time (avg of %d experiments) = %f " % (
    #     num_of_experiments, avg_avg_totalresponse))
    # print("first fit time avg num of requests (avg of %d experiments) = %f " % (
    #     num_of_experiments, ff_time_num_requests / num_of_experiments))
    # file_averages.write('%s,%d,FF TIME,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
    #     config['scenario'], num_of_experiments,
    #     avg_avg_latency,
    #     p50_avg_latency,
    #     p75_avg_latency,
    #     avg_avg_totalresponse,
    #     p50_avg_totalresponse,
    #     p75_avg_totalresponse,
    #     ff_time_num_requests / num_of_experiments,
    #     ff_time_calculation_time / num_of_experiments, ff_time_services_on_fog / num_of_experiments))
    #
    # # memetic w/o lc averages
    # plt.hist(memetic_no_local_search_all_experiments_avg_latency, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_latency_memetic_wo_lc.png')
    # plt.clf()
    #
    # plt.hist(memetic_no_local_search_all_experiments_avg_totalresponse, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_totalresponse_memetic_wo_lc.png')
    # plt.clf()
    #
    # avg_avg_latency = float(np.mean(memetic_no_local_search_all_experiments_avg_latency))
    # p50_avg_latency = float(np.percentile(memetic_no_local_search_all_experiments_avg_latency, 50))
    # p75_avg_latency = float(np.percentile(memetic_no_local_search_all_experiments_avg_latency, 75))
    #
    # avg_avg_totalresponse = float(np.mean(memetic_no_local_search_all_experiments_avg_totalresponse))
    # p50_avg_totalresponse = float(np.percentile(memetic_no_local_search_all_experiments_avg_totalresponse, 50))
    # p75_avg_totalresponse = float(np.percentile(memetic_no_local_search_all_experiments_avg_totalresponse, 75))
    #
    # print("memetic without local search avg latency (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_latency))
    # print("memetic without local search avg total response time (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_totalresponse))
    # print("memetic without local search avg num of requests (avg of %d experiments) = %f " % (
    #     num_of_experiments, memetic_no_local_search_num_requests / num_of_experiments))
    #
    # file_averages.write('%s,%d,memetic w/o LC,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
    #     config['scenario'], num_of_experiments,
    #     avg_avg_latency,
    #     p50_avg_latency,
    #     p75_avg_latency,
    #     avg_avg_totalresponse,
    #     p50_avg_totalresponse,
    #     p75_avg_totalresponse,
    #     memetic_no_local_search_num_requests / num_of_experiments,
    #     memetic_no_local_search_calculation_time / num_of_experiments,
    #     memetic_no_local_search_services_on_fog / num_of_experiments))
    #
    # # memetic experimental averages
    # plt.hist(memetic_experemental_all_experiments_avg_latency, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_latency_memetic_exp.png')
    # plt.clf()
    #
    # plt.hist(memetic_experemental_all_experiments_avg_totalresponse, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_totalresponse_memetic_exp.png')
    # plt.clf()
    #
    # avg_avg_latency = float(np.mean(memetic_experemental_all_experiments_avg_latency))
    # p50_avg_latency = float(np.percentile(memetic_experemental_all_experiments_avg_latency, 50))
    # p75_avg_latency = float(np.percentile(memetic_experemental_all_experiments_avg_latency, 75))
    #
    # avg_avg_totalresponse = float(np.mean(memetic_experemental_all_experiments_avg_totalresponse))
    # p50_avg_totalresponse = float(np.percentile(memetic_experemental_all_experiments_avg_totalresponse, 50))
    # p75_avg_totalresponse = float(np.percentile(memetic_experemental_all_experiments_avg_totalresponse, 75))
    #
    # print("memetic experimental avg latency (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_latency))
    # print("memetic experimental avg total response time (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_totalresponse))
    # print("memetic experimental avg num of requests (avg of %d experiments) = %f " % (
    #     num_of_experiments, memetic_experemental_num_requests / num_of_experiments))
    #
    # file_averages.write('%s,%d,memetic exp 1,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
    #     config['scenario'], num_of_experiments,
    #     avg_avg_latency,
    #     p50_avg_latency,
    #     p75_avg_latency,
    #     avg_avg_totalresponse,
    #     p50_avg_totalresponse,
    #     p75_avg_totalresponse,
    #     memetic_experemental_num_requests / num_of_experiments,
    #     memetic_experimental_calculation_time / num_of_experiments,
    #     memetic_experimental_services_on_fog / num_of_experiments))
    #
    # # memetic experimental 2 averages
    # plt.hist(memetic_experemental2_all_experiments_avg_latency, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_latency_memetic_exp2.png')
    # plt.clf()
    #
    # plt.hist(memetic_experemental2_all_experiments_avg_totalresponse, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_totalresponse_memetic_exp2.png')
    # plt.clf()
    #
    # avg_avg_latency = float(np.mean(memetic_experemental2_all_experiments_avg_latency))
    # p50_avg_latency = float(np.percentile(memetic_experemental2_all_experiments_avg_latency, 50))
    # p75_avg_latency = float(np.percentile(memetic_experemental2_all_experiments_avg_latency, 75))
    #
    # avg_avg_totalresponse = float(np.mean(memetic_experemental2_all_experiments_avg_totalresponse))
    # p50_avg_totalresponse = float(np.percentile(memetic_experemental2_all_experiments_avg_totalresponse, 50))
    # p75_avg_totalresponse = float(np.percentile(memetic_experemental2_all_experiments_avg_totalresponse, 75))
    #
    # print("memetic experimental 2 avg latency (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_latency))
    # print("memetic experimental 2 avg total response time (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_totalresponse))
    # print("memetic experimental 2 avg num of requests (avg of %d experiments) = %f " % (
    #     num_of_experiments, memetic_experemental2_num_requests / num_of_experiments))
    #
    # file_averages.write('%s,%d,memetic exp 2,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
    #     config['scenario'], num_of_experiments,
    #     avg_avg_latency,
    #     p50_avg_latency,
    #     p75_avg_latency,
    #     avg_avg_totalresponse,
    #     p50_avg_totalresponse,
    #     p75_avg_totalresponse,
    #     memetic_experemental2_num_requests / num_of_experiments,
    #     memetic_experimental2_calculation_time / num_of_experiments,
    #     memetic_experimental2_services_on_fog / num_of_experiments))
    #
    # # memetic experimental 3 averages
    # plt.hist(memetic_experemental3_all_experiments_avg_latency, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_latency_memetic_exp3.png')
    # plt.clf()
    #
    # plt.hist(memetic_experemental3_all_experiments_avg_totalresponse, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_totalresponse_memetic_exp3.png')
    # plt.clf()
    #
    # avg_avg_latency = float(np.mean(memetic_experemental3_all_experiments_avg_latency))
    # p50_avg_latency = float(np.percentile(memetic_experemental3_all_experiments_avg_latency, 50))
    # p75_avg_latency = float(np.percentile(memetic_experemental3_all_experiments_avg_latency, 75))
    #
    # avg_avg_totalresponse = float(np.mean(memetic_experemental3_all_experiments_avg_totalresponse))
    # p50_avg_totalresponse = float(np.percentile(memetic_experemental3_all_experiments_avg_totalresponse, 50))
    # p75_avg_totalresponse = float(np.percentile(memetic_experemental3_all_experiments_avg_totalresponse, 75))
    #
    # print("memetic experimental 3 avg latency (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_latency))
    # print("memetic experimental 3 avg total response time (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_totalresponse))
    # print("memetic experimental 3 avg num of requests (avg of %d experiments) = %f " % (
    #     num_of_experiments, memetic_experemental3_num_requests / num_of_experiments))
    #
    # file_averages.write('%s,%d,memetic exp 3,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
    #     config['scenario'], num_of_experiments,
    #     avg_avg_latency,
    #     p50_avg_latency,
    #     p75_avg_latency,
    #     avg_avg_totalresponse,
    #     p50_avg_totalresponse,
    #     p75_avg_totalresponse,
    #     memetic_experemental3_num_requests / num_of_experiments,
    #     memetic_experimental3_calculation_time / num_of_experiments,
    #     memetic_experimental3_services_on_fog / num_of_experiments))
    #
    # # memetic experimental 4 averages
    # plt.hist(memetic_experemental4_all_experiments_avg_latency, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_latency_memetic_exp4.png')
    # plt.clf()
    #
    # plt.hist(memetic_experemental4_all_experiments_avg_totalresponse, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_totalresponse_memetic_exp4.png')
    # plt.clf()
    #
    # avg_avg_latency = float(np.mean(memetic_experemental4_all_experiments_avg_latency))
    # p50_avg_latency = float(np.percentile(memetic_experemental4_all_experiments_avg_latency, 50))
    # p75_avg_latency = float(np.percentile(memetic_experemental4_all_experiments_avg_latency, 75))
    #
    # avg_avg_totalresponse = float(np.mean(memetic_experemental4_all_experiments_avg_totalresponse))
    # p50_avg_totalresponse = float(np.percentile(memetic_experemental4_all_experiments_avg_totalresponse, 50))
    # p75_avg_totalresponse = float(np.percentile(memetic_experemental4_all_experiments_avg_totalresponse, 75))
    #
    # print("memetic experimental 4 avg latency (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_latency))
    # print("memetic experimental 4 avg total response time (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_totalresponse))
    # print("memetic experimental 4 avg num of requests (avg of %d experiments) = %f " % (
    #     num_of_experiments, memetic_experemental4_num_requests / num_of_experiments))
    #
    # file_averages.write('%s,%d,memetic exp 4,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
    #     config['scenario'], num_of_experiments,
    #     avg_avg_latency,
    #     p50_avg_latency,
    #     p75_avg_latency,
    #     avg_avg_totalresponse,
    #     p50_avg_totalresponse,
    #     p75_avg_totalresponse,
    #     memetic_experemental4_num_requests / num_of_experiments,
    #     memetic_experimental4_calculation_time / num_of_experiments,
    #     memetic_experimental4_services_on_fog / num_of_experiments))
    #
    # # memetic experimental 5 averages
    # plt.hist(memetic_experemental5_all_experiments_avg_latency, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_latency_memetic_exp5.png')
    # plt.clf()
    #
    # plt.hist(memetic_experemental5_all_experiments_avg_totalresponse, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_totalresponse_memetic_exp5.png')
    # plt.clf()
    #
    # avg_avg_latency = float(np.mean(memetic_experemental5_all_experiments_avg_latency))
    # p50_avg_latency = float(np.percentile(memetic_experemental5_all_experiments_avg_latency, 50))
    # p75_avg_latency = float(np.percentile(memetic_experemental5_all_experiments_avg_latency, 75))
    #
    # avg_avg_totalresponse = float(np.mean(memetic_experemental5_all_experiments_avg_totalresponse))
    # p50_avg_totalresponse = float(np.percentile(memetic_experemental5_all_experiments_avg_totalresponse, 50))
    # p75_avg_totalresponse = float(np.percentile(memetic_experemental5_all_experiments_avg_totalresponse, 75))
    #
    # print("memetic experimental 5 avg latency (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_latency))
    # print("memetic experimental 5 avg total response time (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_totalresponse))
    # print("memetic experimental 5 avg num of requests (avg of %d experiments) = %f " % (
    #     num_of_experiments, memetic_experemental5_num_requests / num_of_experiments))
    #
    # file_averages.write('%s,%d,memetic exp 5,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
    #     config['scenario'], num_of_experiments,
    #     avg_avg_latency,
    #     p50_avg_latency,
    #     p75_avg_latency,
    #     avg_avg_totalresponse,
    #     p50_avg_totalresponse,
    #     p75_avg_totalresponse,
    #     memetic_experemental5_num_requests / num_of_experiments,
    #     memetic_experimental5_calculation_time / num_of_experiments,
    #     memetic_experimental5_services_on_fog / num_of_experiments))
    #
    # # memetic experimental 6 averages
    # plt.hist(memetic_experemental6_all_experiments_avg_latency, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_latency_memetic_exp6.png')
    # plt.clf()
    #
    # plt.hist(memetic_experemental6_all_experiments_avg_totalresponse, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_totalresponse_memetic_exp6.png')
    # plt.clf()
    #
    # avg_avg_latency = float(np.mean(memetic_experemental6_all_experiments_avg_latency))
    # p50_avg_latency = float(np.percentile(memetic_experemental6_all_experiments_avg_latency, 50))
    # p75_avg_latency = float(np.percentile(memetic_experemental6_all_experiments_avg_latency, 75))
    #
    # avg_avg_totalresponse = float(np.mean(memetic_experemental6_all_experiments_avg_totalresponse))
    # p50_avg_totalresponse = float(np.percentile(memetic_experemental6_all_experiments_avg_totalresponse, 50))
    # p75_avg_totalresponse = float(np.percentile(memetic_experemental6_all_experiments_avg_totalresponse, 75))
    #
    # print("memetic experimental 6 avg latency (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_latency))
    # print("memetic experimental 6 avg total response time (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_totalresponse))
    # print("memetic experimental 6 avg num of requests (avg of %d experiments) = %f " % (
    #     num_of_experiments, memetic_experemental6_num_requests / num_of_experiments))
    #
    # file_averages.write('%s,%d,memetic exp 6,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
    #     config['scenario'], num_of_experiments,
    #     avg_avg_latency,
    #     p50_avg_latency,
    #     p75_avg_latency,
    #     avg_avg_totalresponse,
    #     p50_avg_totalresponse,
    #     p75_avg_totalresponse,
    #     memetic_experemental6_num_requests / num_of_experiments,
    #     memetic_experimental6_calculation_time / num_of_experiments,
    #     memetic_experimental6_services_on_fog / num_of_experiments))
    #
    # # memetic baseline averages
    # plt.hist(memetic_all_experiments_avg_latency, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_latency_memetic.png')
    # plt.clf()
    #
    # plt.hist(memetic_all_experiments_avg_totalresponse, bins=20)
    # plt.savefig(results_folder + 'histograms/' + config['scenario'] + '_totalresponse_memetic.png')
    # plt.clf()
    #
    # avg_avg_latency = float(np.mean(memetic_all_experiments_avg_latency))
    # p50_avg_latency = float(np.percentile(memetic_all_experiments_avg_latency, 50))
    # p75_avg_latency = float(np.percentile(memetic_all_experiments_avg_latency, 75))
    #
    # avg_avg_totalresponse = float(np.mean(memetic_all_experiments_avg_totalresponse))
    # p50_avg_totalresponse = float(np.percentile(memetic_all_experiments_avg_totalresponse, 50))
    # p75_avg_totalresponse = float(np.percentile(memetic_all_experiments_avg_totalresponse, 75))
    #
    # print("memetic avg latency (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_latency))
    # print("memetic avg total response time (avg of %d experiments) = %f" % (
    #     num_of_experiments, avg_avg_totalresponse))
    # print("memetic avg num of requests (avg of %d experiments) = %f " % (
    #     num_of_experiments, memetic_num_requests / num_of_experiments))
    # file_averages.write('%s,%d,memetic,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
    #     config['scenario'], num_of_experiments,
    #     avg_avg_latency,
    #     p50_avg_latency,
    #     p75_avg_latency,
    #     avg_avg_totalresponse,
    #     p50_avg_totalresponse,
    #     p75_avg_totalresponse,
    #     memetic_num_requests / num_of_experiments, memetic_calculation_time / num_of_experiments,
    #     memetic_services_on_fog / num_of_experiments))
    #
    # print("\n\n")
    #
    file.close()
    file_averages.close()
