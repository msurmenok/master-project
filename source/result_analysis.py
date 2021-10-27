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
    # simulation_time = 10000
    simulation_time = 100000

    # to calculate average of time latency over all experiments for this scenario
    ff_ram_all_experiments_avg_latency = 0
    ff_time_all_experiments_avg_latency = 0
    memetic_no_local_search_all_experiments_avg_latency = 0
    memetic_experemental_all_experiments_avg_latency = 0
    memetic_experemental2_all_experiments_avg_latency = 0
    memetic_experemental3_all_experiments_avg_latency = 0
    memetic_experemental4_all_experiments_avg_latency = 0
    memetic_experemental5_all_experiments_avg_latency = 0
    memetic_experemental6_all_experiments_avg_latency = 0
    memetic_all_experiments_avg_latency = 0

    ff_ram_all_experiments_avg_totalresponse = 0
    ff_time_all_experiments_avg_totalresponse = 0
    memetic_no_local_search_all_experiments_avg_totalresponse = 0
    memetic_experemental_all_experiments_avg_totalresponse = 0
    memetic_experemental2_all_experiments_avg_totalresponse = 0
    memetic_experemental3_all_experiments_avg_totalresponse = 0
    memetic_experemental4_all_experiments_avg_totalresponse = 0
    memetic_experemental5_all_experiments_avg_totalresponse = 0
    memetic_experemental6_all_experiments_avg_totalresponse = 0
    memetic_all_experiments_avg_totalresponse = 0

    ff_ram_num_requests = 0
    ff_time_num_requests = 0
    memetic_no_local_search_num_requests = 0
    memetic_experemental_num_requests = 0
    memetic_experemental2_num_requests = 0
    memetic_experemental3_num_requests = 0
    memetic_experemental4_num_requests = 0
    memetic_experemental5_num_requests = 0
    memetic_experemental6_num_requests = 0
    memetic_num_requests = 0

    ff_ram_calculation_time = 0
    ff_time_calculation_time = 0
    memetic_no_local_search_calculation_time = 0
    memetic_experimental_calculation_time = 0
    memetic_experimental2_calculation_time = 0
    memetic_experimental3_calculation_time = 0
    memetic_experimental4_calculation_time = 0
    memetic_experimental5_calculation_time = 0
    memetic_experimental6_calculation_time = 0
    memetic_calculation_time = 0

    ff_ram_services_on_fog = 0
    ff_time_services_on_fog = 0
    memetic_no_local_search_services_on_fog = 0
    memetic_experimental_services_on_fog = 0
    memetic_experimental2_services_on_fog = 0
    memetic_experimental3_services_on_fog = 0
    memetic_experimental4_services_on_fog = 0
    memetic_experimental5_services_on_fog = 0
    memetic_experimental6_services_on_fog = 0
    memetic_services_on_fog = 0

    num_of_experiments = config['iterations']
    results_folder = "results/current/"

    results_analysis_file = results_folder + "analysis.csv"
    results_analysis_average_file = results_folder + "analysis_averages.csv"

    file = open(results_analysis_file, 'a+')  # save completion time
    file.write(
        'scenario,# experiment,algorithm,mean latency,min latency,max latency,average total response,num requests\n')
    file_averages = open(results_analysis_average_file, 'a+')
    file_averages.write(
        'scenario,# experiments,algorithm,average latency,average total response,average num requests,average calculation time,average num services on fog\n')

    for i in range(num_of_experiments):
        # calculate total time taken by each algorithm:
        algorithm_time_filepath = results_folder + 'results_' + config['scenario'] + '_' + str(
            i) + '/algorithm_time.csv'

        data = pd.read_csv(algorithm_time_filepath, header=None)
        algorithm_to_time = list(zip(data[1].to_list(), data[2].to_list(), data[3].to_list()))
        for tuple in algorithm_to_time:
            if tuple[0] == 'FirstFitRAM':
                ff_ram_calculation_time += tuple[1]
                ff_ram_services_on_fog += tuple[2]
            elif tuple[0] == 'FirstFitTime':
                ff_time_calculation_time += tuple[1]
                ff_time_services_on_fog += tuple[2]
            elif tuple[0] == 'MemeticWithoutLocalSearch':
                memetic_no_local_search_calculation_time += tuple[1]
                memetic_no_local_search_services_on_fog += tuple[2]
            elif tuple[0] == 'MemeticExperimental1':
                memetic_experimental_calculation_time += tuple[1]
                memetic_experimental_services_on_fog += tuple[2]
            elif tuple[0] == 'MemeticExperimental2':
                memetic_experimental2_calculation_time += tuple[1]
                memetic_experimental2_services_on_fog += tuple[2]
            elif tuple[0] == 'MemeticExperimental3':
                memetic_experimental3_calculation_time += tuple[1]
                memetic_experimental3_services_on_fog += tuple[2]
            elif tuple[0] == 'MemeticExperimental4':
                memetic_experimental4_calculation_time += tuple[1]
                memetic_experimental4_services_on_fog += tuple[2]
            elif tuple[0] == 'MemeticExperimental5':
                memetic_experimental5_calculation_time += tuple[1]
                memetic_experimental5_services_on_fog += tuple[2]
            elif tuple[0] == 'MemeticExperimental6':
                memetic_experimental6_calculation_time += tuple[1]
                memetic_experimental6_services_on_fog += tuple[2]
            elif tuple[0] == 'Memetic':
                memetic_calculation_time += tuple[1]
                memetic_services_on_fog += tuple[2]

        # First Fit by RAM
        print("--------- %d Simulation Results ----------" % i)

        df_firstfit_ram = pd.read_csv(
            (results_folder + "/results_%s_%dResults_FirstFitRAM_%s_%d_%d.csv") % (
                config['scenario'], i, config['scenario'], simulation_time, i))
        compute_times_df(df_firstfit_ram)

        ff_ram_all_experiments_avg_latency += df_firstfit_ram["time_latency"].mean()
        ff_ram_all_experiments_avg_totalresponse += df_firstfit_ram["time_total_response"].mean()
        ff_ram_num_requests += len(df_firstfit_ram)

        print("FF RAM number of tasks processed: %d" % len(df_firstfit_ram))
        print("FF RAM average latency", df_firstfit_ram["time_latency"].mean())
        print("FF RAM minimum latency", df_firstfit_ram["time_latency"].min())
        print("FF RAM maximum latency", df_firstfit_ram["time_latency"].max())

        print("FF RAM total time total response", df_firstfit_ram["time_total_response"].sum())
        print("FF RAM mean time total response", df_firstfit_ram["time_total_response"].mean())
        print()

        file.write('%s,experiment_%d,FF RAM,%f,%f,%f,%f,%d\n' % (
            config['scenario'], i, df_firstfit_ram["time_latency"].mean(), df_firstfit_ram["time_latency"].min(),
            df_firstfit_ram["time_latency"].max(), df_firstfit_ram["time_total_response"].mean(), len(df_firstfit_ram)))

        # First Fit by Time
        df_firstfit_time = pd.read_csv(
            (results_folder + "/results_%s_%dResults_FirstFitTime_%s_%d_%d.csv") % (
                config['scenario'], i, config['scenario'], simulation_time, i))
        compute_times_df(df_firstfit_time)
        ff_time_all_experiments_avg_latency += df_firstfit_time["time_latency"].mean()
        ff_time_all_experiments_avg_totalresponse += df_firstfit_time["time_total_response"].mean()
        ff_time_num_requests += len(df_firstfit_time)

        print("FF Time number of tasks processed: %d" % len(df_firstfit_time))
        print("FF Time average latency", df_firstfit_time["time_latency"].mean())
        print("FF Time minimum latency", df_firstfit_time["time_latency"].min())
        print("FF Time maximum latency", df_firstfit_time["time_latency"].max())

        print("FF Time total time total response", df_firstfit_time["time_total_response"].sum())
        print("FF Time mean time total response", df_firstfit_time["time_total_response"].mean())
        print()

        file.write('%s,experiment_%d,FF Time,%f,%f,%f,%f,%d\n' % (
            config['scenario'], i, df_firstfit_time["time_latency"].mean(), df_firstfit_time["time_latency"].min(),
            df_firstfit_time["time_latency"].max(), df_firstfit_time["time_total_response"].mean(),
            len(df_firstfit_time)))

        # Memetic without Local Search
        df_memetic_no_local_search = pd.read_csv(
            (results_folder + "/results_%s_%dResults_MemeticWithoutLocalSearch_%s_%d_%d.csv") % (
                config['scenario'], i, config['scenario'], simulation_time, i))
        compute_times_df(df_memetic_no_local_search)

        memetic_no_local_search_all_experiments_avg_latency += df_memetic_no_local_search["time_latency"].mean()
        memetic_no_local_search_all_experiments_avg_totalresponse += df_memetic_no_local_search[
            "time_total_response"].mean()
        memetic_no_local_search_num_requests += len(df_memetic_no_local_search)

        print("Memetic w/o LC number of tasks processed: %d" % len(df_memetic_no_local_search))
        print("Memetic w/o LC average latency", df_memetic_no_local_search["time_latency"].mean())
        print("Memetic w/o LC minimum latency", df_memetic_no_local_search["time_latency"].min())
        print("Memetic w/o LC maximum latency", df_memetic_no_local_search["time_latency"].max())

        print("Memetic w/o LC total time total response", df_memetic_no_local_search["time_total_response"].sum())
        print("Memetic w/o LC mean time total response", df_memetic_no_local_search["time_total_response"].mean())
        print()

        file.write('%s,experiment_%d,Memetic w/o LC,%f,%f,%f,%f,%d\n' % (
            config['scenario'], i, df_memetic_no_local_search["time_latency"].mean(),
            df_memetic_no_local_search["time_latency"].min(),
            df_memetic_no_local_search["time_latency"].max(), df_memetic_no_local_search["time_total_response"].mean(),
            len(df_memetic_no_local_search)))

        # Memetic Experimental
        df_memetic_experimental = pd.read_csv(
            (results_folder + "/results_%s_%dResults_MemeticExperimental_%s_%d_%d.csv") % (
                config['scenario'], i, config['scenario'], simulation_time, i))
        compute_times_df(df_memetic_experimental)

        memetic_experemental_all_experiments_avg_latency += df_memetic_experimental["time_latency"].mean()
        memetic_experemental_all_experiments_avg_totalresponse += df_memetic_experimental["time_total_response"].mean()
        memetic_experemental_num_requests += len(df_memetic_experimental)

        print("Memetic Experimental number of tasks processed: %d" % len(df_memetic_experimental))
        print("Memetic Experimental average latency", df_memetic_experimental["time_latency"].mean())
        print("Memetic Experimental minimum latency", df_memetic_experimental["time_latency"].min())
        print("Memetic Experimental maximum latency", df_memetic_experimental["time_latency"].max())

        print("Memetic Experimental total time total response", df_memetic_experimental["time_total_response"].sum())
        print("Memetic Experimental mean time total response", df_memetic_experimental["time_total_response"].mean())
        print()

        file.write('%s,experiment_%d,Memetic Experimental,%f,%f,%f,%f,%d\n' % (
            config['scenario'], i, df_memetic_experimental["time_latency"].mean(),
            df_memetic_experimental["time_latency"].min(),
            df_memetic_experimental["time_latency"].max(), df_memetic_experimental["time_total_response"].mean(),
            len(df_memetic_experimental)))

        # Memetic Experimental 2
        df_memetic_experimental2 = pd.read_csv(
            (results_folder + "/results_%s_%dResults_MemeticExperimental2_%s_%d_%d.csv") % (
                config['scenario'], i, config['scenario'], simulation_time, i))
        compute_times_df(df_memetic_experimental2)

        memetic_experemental2_all_experiments_avg_latency += df_memetic_experimental2["time_latency"].mean()
        memetic_experemental2_all_experiments_avg_totalresponse += df_memetic_experimental2[
            "time_total_response"].mean()
        memetic_experemental2_num_requests += len(df_memetic_experimental2)

        print("Memetic Experimental 2 number of tasks processed: %d" % len(df_memetic_experimental2))
        print("Memetic Experimental 2 average latency", df_memetic_experimental2["time_latency"].mean())
        print("Memetic Experimental 2 minimum latency", df_memetic_experimental2["time_latency"].min())
        print("Memetic Experimental 2 maximum latency", df_memetic_experimental2["time_latency"].max())

        print("Memetic Experimental 2 total time total response", df_memetic_experimental2["time_total_response"].sum())
        print("Memetic Experimental 2 mean time total response", df_memetic_experimental2["time_total_response"].mean())
        print()

        file.write('%s,experiment_%d,Memetic Experimental 2,%f,%f,%f,%f,%d\n' % (
            config['scenario'], i, df_memetic_experimental2["time_latency"].mean(),
            df_memetic_experimental2["time_latency"].min(),
            df_memetic_experimental2["time_latency"].max(), df_memetic_experimental2["time_total_response"].mean(),
            len(df_memetic_experimental2)))

        # Memetic Experimental 3
        df_memetic_experimental3 = pd.read_csv(
            (results_folder + "/results_%s_%dResults_MemeticExperimental3_%s_%d_%d.csv") % (
                config['scenario'], i, config['scenario'], simulation_time, i))
        compute_times_df(df_memetic_experimental3)

        memetic_experemental3_all_experiments_avg_latency += df_memetic_experimental3["time_latency"].mean()
        memetic_experemental3_all_experiments_avg_totalresponse += df_memetic_experimental3[
            "time_total_response"].mean()
        memetic_experemental3_num_requests += len(df_memetic_experimental3)

        print("Memetic Experimental 3 number of tasks processed: %d" % len(df_memetic_experimental3))
        print("Memetic Experimental 3 average latency", df_memetic_experimental3["time_latency"].mean())
        print("Memetic Experimental 3 minimum latency", df_memetic_experimental3["time_latency"].min())
        print("Memetic Experimental 3 maximum latency", df_memetic_experimental3["time_latency"].max())

        print("Memetic Experimental 3 total time total response", df_memetic_experimental3["time_total_response"].sum())
        print("Memetic Experimental 3 mean time total response", df_memetic_experimental3["time_total_response"].mean())
        print()

        file.write('%s,experiment_%d,Memetic Experimental 3,%f,%f,%f,%f,%d\n' % (
            config['scenario'], i, df_memetic_experimental3["time_latency"].mean(),
            df_memetic_experimental3["time_latency"].min(),
            df_memetic_experimental3["time_latency"].max(), df_memetic_experimental3["time_total_response"].mean(),
            len(df_memetic_experimental3)))

        # Memetic Experimental 4
        df_memetic_experimental4 = pd.read_csv(
            (results_folder + "/results_%s_%dResults_MemeticExperimental4_%s_%d_%d.csv") % (
                config['scenario'], i, config['scenario'], simulation_time, i))
        compute_times_df(df_memetic_experimental4)

        memetic_experemental4_all_experiments_avg_latency += df_memetic_experimental4["time_latency"].mean()
        memetic_experemental4_all_experiments_avg_totalresponse += df_memetic_experimental4[
            "time_total_response"].mean()
        memetic_experemental4_num_requests += len(df_memetic_experimental4)

        print("Memetic Experimental 4 number of tasks processed: %d" % len(df_memetic_experimental4))
        print("Memetic Experimental 4 average latency", df_memetic_experimental4["time_latency"].mean())
        print("Memetic Experimental 4 minimum latency", df_memetic_experimental4["time_latency"].min())
        print("Memetic Experimental 4 maximum latency", df_memetic_experimental4["time_latency"].max())

        print("Memetic Experimental 4 total time total response", df_memetic_experimental4["time_total_response"].sum())
        print("Memetic Experimental 4 mean time total response", df_memetic_experimental4["time_total_response"].mean())
        print()

        file.write('%s,experiment_%d,Memetic Experimental 4,%f,%f,%f,%f,%d\n' % (
            config['scenario'], i, df_memetic_experimental4["time_latency"].mean(),
            df_memetic_experimental4["time_latency"].min(),
            df_memetic_experimental4["time_latency"].max(), df_memetic_experimental4["time_total_response"].mean(),
            len(df_memetic_experimental4)))

        # Memetic Experimental 5
        df_memetic_experimental5 = pd.read_csv(
            (results_folder + "/results_%s_%dResults_MemeticExperimental5_%s_%d_%d.csv") % (
                config['scenario'], i, config['scenario'], simulation_time, i))
        compute_times_df(df_memetic_experimental5)

        memetic_experemental5_all_experiments_avg_latency += df_memetic_experimental5["time_latency"].mean()
        memetic_experemental5_all_experiments_avg_totalresponse += df_memetic_experimental5[
            "time_total_response"].mean()
        memetic_experemental5_num_requests += len(df_memetic_experimental5)

        print("Memetic Experimental 5 number of tasks processed: %d" % len(df_memetic_experimental5))
        print("Memetic Experimental 5 average latency", df_memetic_experimental5["time_latency"].mean())
        print("Memetic Experimental 5 minimum latency", df_memetic_experimental5["time_latency"].min())
        print("Memetic Experimental 5 maximum latency", df_memetic_experimental5["time_latency"].max())

        print("Memetic Experimental 5 total time total response", df_memetic_experimental5["time_total_response"].sum())
        print("Memetic Experimental 5 mean time total response", df_memetic_experimental5["time_total_response"].mean())
        print()

        file.write('%s,experiment_%d,Memetic Experimental 5,%f,%f,%f,%f,%d\n' % (
            config['scenario'], i, df_memetic_experimental5["time_latency"].mean(),
            df_memetic_experimental5["time_latency"].min(),
            df_memetic_experimental5["time_latency"].max(), df_memetic_experimental5["time_total_response"].mean(),
            len(df_memetic_experimental5)))

        # Memetic Experimental 6
        df_memetic_experimental6 = pd.read_csv(
            (results_folder + "/results_%s_%dResults_MemeticExperimental6_%s_%d_%d.csv") % (
                config['scenario'], i, config['scenario'], simulation_time, i))
        compute_times_df(df_memetic_experimental6)

        memetic_experemental6_all_experiments_avg_latency += df_memetic_experimental6["time_latency"].mean()
        memetic_experemental6_all_experiments_avg_totalresponse += df_memetic_experimental6[
            "time_total_response"].mean()
        memetic_experemental6_num_requests += len(df_memetic_experimental6)

        print("Memetic Experimental 6 number of tasks processed: %d" % len(df_memetic_experimental6))
        print("Memetic Experimental 6 average latency", df_memetic_experimental6["time_latency"].mean())
        print("Memetic Experimental 6 minimum latency", df_memetic_experimental6["time_latency"].min())
        print("Memetic Experimental 6 maximum latency", df_memetic_experimental6["time_latency"].max())

        print("Memetic Experimental 6 total time total response", df_memetic_experimental6["time_total_response"].sum())
        print("Memetic Experimental 6 mean time total response", df_memetic_experimental6["time_total_response"].mean())
        print()

        file.write('%s,experiment_%d,Memetic Experimental 6,%f,%f,%f,%f,%d\n' % (
            config['scenario'], i, df_memetic_experimental6["time_latency"].mean(),
            df_memetic_experimental6["time_latency"].min(),
            df_memetic_experimental6["time_latency"].max(), df_memetic_experimental6["time_total_response"].mean(),
            len(df_memetic_experimental6)))


        # Memetic (with local search)
        df_memetic = pd.read_csv(
            (results_folder + "/results_%s_%dResults_Memetic_%s_%d_%d.csv") % (
                config['scenario'], i, config['scenario'], simulation_time, i))
        compute_times_df(df_memetic)

        memetic_all_experiments_avg_latency += df_memetic["time_latency"].mean()
        memetic_all_experiments_avg_totalresponse += df_memetic["time_total_response"].mean()
        memetic_num_requests += len(df_memetic)

        print("Memetic number of tasks processed: %d" % len(df_memetic))
        print("Memetic average latency", df_memetic["time_latency"].mean())
        print("Memetic minimum latency", df_memetic["time_latency"].min())
        print("Memetic maximum latency", df_memetic["time_latency"].max())

        print("Memetic total time total response", df_memetic["time_total_response"].sum())
        print("Memetic mean time total response", df_memetic["time_total_response"].mean())
        print()
        file.write('%s,experiment_%d,Memetic,%f,%f,%f,%f,%d\n' % (
            config['scenario'], i, df_memetic["time_latency"].mean(), df_memetic["time_latency"].min(),
            df_memetic["time_latency"].max(), df_memetic["time_total_response"].mean(), len(df_memetic)))

    print("=============== %d experiments average =====================" % num_of_experiments)
    print("first fit ram avg latency (avg of %d experiments) = %f " % (
        num_of_experiments, ff_ram_all_experiments_avg_latency / num_of_experiments))
    print("first fit ram avg total response time (avg of %d experiments) = %f " % (
        num_of_experiments, ff_ram_all_experiments_avg_totalresponse / num_of_experiments))
    print("first fit ram avg num of requests (avg of %d experiments) = %f " % (
        num_of_experiments, ff_ram_num_requests / num_of_experiments))

    file_averages.write('%s,%d,FF RAM,%f,%f,%f,%f,%f\n' % (
        config['scenario'], num_of_experiments, ff_ram_all_experiments_avg_latency / num_of_experiments,
        ff_ram_all_experiments_avg_totalresponse / num_of_experiments, ff_ram_num_requests / num_of_experiments,
        ff_ram_calculation_time / num_of_experiments, ff_ram_services_on_fog / num_of_experiments))

    # first fit time averages
    print("first fit time avg latency (avg of %d experiments) = %f " % (
        num_of_experiments, ff_time_all_experiments_avg_latency / num_of_experiments))
    print("first fit time avg total response time (avg of %d experiments) = %f " % (
        num_of_experiments, ff_time_all_experiments_avg_totalresponse / num_of_experiments))
    print("first fit time avg num of requests (avg of %d experiments) = %f " % (
        num_of_experiments, ff_time_num_requests / num_of_experiments))
    file_averages.write('%s,%d,FF TIME,%f,%f,%f,%f,%f\n' % (
        config['scenario'], num_of_experiments, ff_time_all_experiments_avg_totalresponse / num_of_experiments,
        ff_time_all_experiments_avg_totalresponse / num_of_experiments, ff_time_num_requests / num_of_experiments,
        ff_time_calculation_time / num_of_experiments, ff_time_services_on_fog / num_of_experiments))

    # memetic w/o lc averages
    print("memetic without local search avg latency (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_no_local_search_all_experiments_avg_latency / num_of_experiments))
    print("memetic without local search avg total response time (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_no_local_search_all_experiments_avg_totalresponse / num_of_experiments))
    print("memetic without local search avg num of requests (avg of %d experiments) = %f " % (
        num_of_experiments, memetic_no_local_search_num_requests / num_of_experiments))

    file_averages.write('%s,%d,memetic w/o LC,%f,%f,%f,%f,%f\n' % (
        config['scenario'], num_of_experiments,
        memetic_no_local_search_all_experiments_avg_latency / num_of_experiments,
        memetic_no_local_search_all_experiments_avg_totalresponse / num_of_experiments,
        memetic_no_local_search_num_requests / num_of_experiments,
        memetic_no_local_search_calculation_time / num_of_experiments,
        memetic_no_local_search_services_on_fog / num_of_experiments))

    # memetic experimental averages
    print("memetic experimental avg latency (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_experemental_all_experiments_avg_latency / num_of_experiments))
    print("memetic experimental avg total response time (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_experemental_all_experiments_avg_totalresponse / num_of_experiments))
    print("memetic experimental avg num of requests (avg of %d experiments) = %f " % (
        num_of_experiments, memetic_experemental_num_requests / num_of_experiments))

    file_averages.write('%s,%d,memetic exp 1,%f,%f,%f,%f,%f\n' % (
        config['scenario'], num_of_experiments,
        memetic_experemental_all_experiments_avg_latency / num_of_experiments,
        memetic_experemental_all_experiments_avg_totalresponse / num_of_experiments,
        memetic_experemental_num_requests / num_of_experiments,
        memetic_experimental_calculation_time / num_of_experiments,
        memetic_experimental_services_on_fog / num_of_experiments))

    # memetic experimental 2 averages
    print("memetic experimental 2 avg latency (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_experemental2_all_experiments_avg_latency / num_of_experiments))
    print("memetic experimental 2 avg total response time (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_experemental2_all_experiments_avg_totalresponse / num_of_experiments))
    print("memetic experimental 2 avg num of requests (avg of %d experiments) = %f " % (
        num_of_experiments, memetic_experemental2_num_requests / num_of_experiments))

    file_averages.write('%s,%d,memetic exp 2,%f,%f,%f,%f,%f\n' % (
        config['scenario'], num_of_experiments,
        memetic_experemental2_all_experiments_avg_latency / num_of_experiments,
        memetic_experemental2_all_experiments_avg_totalresponse / num_of_experiments,
        memetic_experemental2_num_requests / num_of_experiments,
        memetic_experimental2_calculation_time / num_of_experiments,
        memetic_experimental2_services_on_fog / num_of_experiments))

    # memetic experimental 3 averages
    print("memetic experimental 3 avg latency (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_experemental3_all_experiments_avg_latency / num_of_experiments))
    print("memetic experimental 3 avg total response time (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_experemental3_all_experiments_avg_totalresponse / num_of_experiments))
    print("memetic experimental 3 avg num of requests (avg of %d experiments) = %f " % (
        num_of_experiments, memetic_experemental3_num_requests / num_of_experiments))

    file_averages.write('%s,%d,memetic exp 3,%f,%f,%f,%f,%f\n' % (
        config['scenario'], num_of_experiments,
        memetic_experemental3_all_experiments_avg_latency / num_of_experiments,
        memetic_experemental3_all_experiments_avg_totalresponse / num_of_experiments,
        memetic_experemental3_num_requests / num_of_experiments,
        memetic_experimental3_calculation_time / num_of_experiments,
        memetic_experimental3_services_on_fog / num_of_experiments))

    # memetic experimental 4 averages
    print("memetic experimental 4 avg latency (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_experemental4_all_experiments_avg_latency / num_of_experiments))
    print("memetic experimental 4 avg total response time (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_experemental4_all_experiments_avg_totalresponse / num_of_experiments))
    print("memetic experimental 4 avg num of requests (avg of %d experiments) = %f " % (
        num_of_experiments, memetic_experemental4_num_requests / num_of_experiments))

    file_averages.write('%s,%d,memetic exp 4,%f,%f,%f,%f,%f\n' % (
        config['scenario'], num_of_experiments,
        memetic_experemental4_all_experiments_avg_latency / num_of_experiments,
        memetic_experemental4_all_experiments_avg_totalresponse / num_of_experiments,
        memetic_experemental4_num_requests / num_of_experiments,
        memetic_experimental4_calculation_time / num_of_experiments,
        memetic_experimental4_services_on_fog / num_of_experiments))

    # memetic experimental 5 averages
    print("memetic experimental 5 avg latency (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_experemental5_all_experiments_avg_latency / num_of_experiments))
    print("memetic experimental 5 avg total response time (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_experemental5_all_experiments_avg_totalresponse / num_of_experiments))
    print("memetic experimental 5 avg num of requests (avg of %d experiments) = %f " % (
        num_of_experiments, memetic_experemental5_num_requests / num_of_experiments))

    file_averages.write('%s,%d,memetic exp 5,%f,%f,%f,%f,%f\n' % (
        config['scenario'], num_of_experiments,
        memetic_experemental5_all_experiments_avg_latency / num_of_experiments,
        memetic_experemental5_all_experiments_avg_totalresponse / num_of_experiments,
        memetic_experemental5_num_requests / num_of_experiments,
        memetic_experimental5_calculation_time / num_of_experiments,
        memetic_experimental5_services_on_fog / num_of_experiments))

    # memetic experimental 6 averages
    print("memetic experimental 6 avg latency (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_experemental6_all_experiments_avg_latency / num_of_experiments))
    print("memetic experimental 6 avg total response time (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_experemental6_all_experiments_avg_totalresponse / num_of_experiments))
    print("memetic experimental 6 avg num of requests (avg of %d experiments) = %f " % (
        num_of_experiments, memetic_experemental6_num_requests / num_of_experiments))

    file_averages.write('%s,%d,memetic exp 6,%f,%f,%f,%f,%f\n' % (
        config['scenario'], num_of_experiments,
        memetic_experemental6_all_experiments_avg_latency / num_of_experiments,
        memetic_experemental6_all_experiments_avg_totalresponse / num_of_experiments,
        memetic_experemental6_num_requests / num_of_experiments,
        memetic_experimental6_calculation_time / num_of_experiments,
        memetic_experimental6_services_on_fog / num_of_experiments))

    # memetic baseline averages
    print("memetic avg latency (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_all_experiments_avg_latency / num_of_experiments))
    print("memetic avg total response time (avg of %d experiments) = %f" % (
        num_of_experiments, memetic_all_experiments_avg_totalresponse / num_of_experiments))
    print("memetic avg num of requests (avg of %d experiments) = %f " % (
        num_of_experiments, memetic_num_requests / num_of_experiments))
    file_averages.write('%s,%d,memetic,%f,%f,%f,%f,%f\n' % (
        config['scenario'], num_of_experiments,
        memetic_all_experiments_avg_latency / num_of_experiments,
        memetic_all_experiments_avg_totalresponse / num_of_experiments,
        memetic_num_requests / num_of_experiments, memetic_calculation_time / num_of_experiments,
        memetic_services_on_fog / num_of_experiments))

    print("\n\n")

    file.close()
    file_averages.close()
