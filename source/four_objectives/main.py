from functools import partial
import os
import time
import json
import random
import logging.config
from ExperimentSetup import ExperimentSetup
from ExperimentConfigs import configs
import numpy as np
from multiprocessing import Pool

import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt

from yafs.core import Sim
from yafs.application import create_applications_from_json
from yafs.topology import Topology

from yafs.placement import JSONPlacement
from yafs.path_routing import DeviceSpeedAwareRouting
from yafs.distribution import deterministic_distribution
from jsonPopulation import JSONPopulation


# folder_results = "results/"


def main(stop_time, it, algorithm, config, folder_results, folder_data):
    # Create topology from json
    # folder_data = '/' + folder_data
    topo = Topology()
    topology_json = json.load(open(folder_data + "/netDefinition.json"))
    # topo.load(topology_json)
    topo.load_all_node_attr(topology_json)
    # topo.write("data_net.gexf")

    # create applications
    data_app = json.load(open(folder_data + "/appDefinition.json"))
    apps = create_applications_from_json(data_app)

    # load placement algorithm
    placementJson = json.load(open(folder_data + "/allocDefinition" + algorithm + ".json"))
    placement = JSONPlacement(name="Placement", json=placementJson)

    # open(os.path.dirname(__file__) + '/' + self.resultFolder + "/netDefinition.json", "w")
    # load population
    dataPopulation = json.load(open(folder_data + "/usersDefinition.json"))
    pop = JSONPopulation(name="Statical", json=dataPopulation, iteration=it)

    # Routing algorithm
    # selectorPath = MinimunPath()
    selectorPath = DeviceSpeedAwareRouting()
    # folder_results = 'results/'
    s = Sim(topo, default_results_path=folder_results + "Results_%s_%s_%i_%i" % (
        algorithm, config['scenario'], stop_time, it))

    for aName in apps.keys():
        print("Deploying app: ", aName)
        pop_app = JSONPopulation(name="Statical_%s" % aName, json={}, iteration=it)
        data = []
        for element in pop.data["sources"]:
            if element['app'] == aName:
                data.append(element)
        pop_app.data["sources"] = data
        s.deploy_app2(apps[aName], placement, pop_app, selectorPath)

    s.run(stop_time, show_progress_monitor=False)


def initialize_experiment(config, iteration, folder_results, folder_data):
    sg = ExperimentSetup(config=config, folder_data=folder_data)
    sg.networkGeneration()
    sg.appGeneration()
    sg.userGeneration()

    # First Fit RAM
    # start_time = time.time()  # measure time for placement
    # services_placement_count = sg.firstFitRAMPlacement()
    # finish_time = time.time() - start_time
    #
    # services_in_fog, services_in_cloud = services_placement_count
    # file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    # file.write('%s,FirstFitRAM,%s,%s,%s\n' % (
    #     config['scenario'] + '_' + str(iteration), str(finish_time), str(services_in_fog), str(services_in_cloud)))
    #
    # # First Fit TIME
    # start_time = time.time()  # measure time for placement
    # services_placement_count = sg.firstFitTimePlacement()
    # finish_time = time.time() - start_time
    #
    # services_in_fog, services_in_cloud = services_placement_count
    # file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    # file.write('%s,FirstFitTime,%s,%s,%s\n' % (
    #     config['scenario'] + '_' + str(iteration), str(finish_time), str(services_in_fog), str(services_in_cloud)))

    # Memetic Algorithm
    num_creatures = 100
    num_generations = 1000

    # Memetic Algorithm without Local Search
    # start_time = time.time()  # measure time to complete
    # services_placement_count = sg.memeticWithoutLocalSearchPlacement(num_creatures, num_generations)
    # finish_time = time.time() - start_time
    #
    # services_in_fog, services_in_cloud = services_placement_count
    # file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    # file.write('%s,MemeticWithoutLocalSearch,%s,%s,%s\n' % (
    #     config['scenario'] + '_' + str(iteration), str(finish_time), str(services_in_fog), str(services_in_cloud)))

    # Memetic experimental
    start_time = time.time()  # measure time to complete
    services_placement_count = sg.memeticExperimentalPlacement(num_creatures, num_generations)
    finish_time = time.time() - start_time

    services_in_fog, services_in_cloud = services_placement_count
    file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    file.write('%s,MemeticExperimental1,%s,%s,%s\n' % (
        config['scenario'] + '_' + str(iteration), str(finish_time), str(services_in_fog), str(services_in_cloud)))

    # Memetic experimental 3
    # start_time = time.time()  # measure time to complete
    # services_placement_count = sg.memeticExperimentalPlacement3(num_creatures, num_generations)
    # finish_time = time.time() - start_time
    #
    # services_in_fog, services_in_cloud = services_placement_count
    # file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    # file.write('%s,MemeticExperimental3,%s,%s,%s\n' % (
    #     config['scenario'] + '_' + str(iteration), str(finish_time), str(services_in_fog), str(services_in_cloud)))

    # Memetic experimental 2
    start_time = time.time()  # measure time to complete
    services_placement_count = sg.memeticExperimentalPlacement2(num_creatures, num_generations)
    finish_time = time.time() - start_time

    services_in_fog, services_in_cloud = services_placement_count
    file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    file.write('%s,MemeticExperimental2,%s,%s,%s\n' % (
        config['scenario'] + '_' + str(iteration), str(finish_time), str(services_in_fog), str(services_in_cloud)))

    # # Memetic experimental 4
    # start_time = time.time()  # measure time to complete
    # services_placement_count = sg.memeticExperimentalPlacement4(num_creatures, num_generations)
    # finish_time = time.time() - start_time
    #
    # services_in_fog, services_in_cloud = services_placement_count
    # file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    # file.write('%s,MemeticExperimental4,%s,%s,%s\n' % (
    #     config['scenario'] + '_' + str(iteration), str(finish_time), str(services_in_fog), str(services_in_cloud)))
    #
    # # Memetic experimental 5
    # start_time = time.time()  # measure time to complete
    # services_placement_count = sg.memeticExperimentalPlacement5(num_creatures, num_generations)
    # finish_time = time.time() - start_time
    #
    # services_in_fog, services_in_cloud = services_placement_count
    # file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    # file.write('%s,MemeticExperimental5,%s,%s,%s\n' % (
    #     config['scenario'] + '_' + str(iteration), str(finish_time), str(services_in_fog), str(services_in_cloud)))

    # Memetic experimental 6
    start_time = time.time()  # measure time to complete
    services_placement_count = sg.memeticExperimentalPlacement6(num_creatures, num_generations)
    finish_time = time.time() - start_time

    services_in_fog, services_in_cloud = services_placement_count
    file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    file.write('%s,MemeticExperimental6,%s,%s,%s\n' % (
        config['scenario'] + '_' + str(iteration), str(finish_time), str(services_in_fog), str(services_in_cloud)))

    # Memetic experimental 7
    start_time = time.time()  # measure time to complete
    services_placement_count = sg.memeticExperimentalPlacement7(num_creatures, num_generations)
    finish_time = time.time() - start_time

    services_in_fog, services_in_cloud = services_placement_count
    file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    file.write('%s,MemeticExperimental7,%s,%s,%s\n' % (
        config['scenario'] + '_' + str(iteration), str(finish_time), str(services_in_fog), str(services_in_cloud)))

    # Memetic experimental 8
    start_time = time.time()  # measure time to complete
    services_placement_count = sg.memeticExperimentalPlacement8(num_creatures, num_generations)
    finish_time = time.time() - start_time

    services_in_fog, services_in_cloud = services_placement_count
    file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    file.write('%s,MemeticExperimental8,%s,%s,%s\n' % (
        config['scenario'] + '_' + str(iteration), str(finish_time), str(services_in_fog), str(services_in_cloud)))

    # Memetic experimental 9
    # start_time = time.time()  # measure time to complete
    # services_placement_count = sg.memeticExperimentalPlacement9(num_creatures, num_generations)
    # finish_time = time.time() - start_time
    #
    # services_in_fog, services_in_cloud = services_placement_count
    # file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    # file.write('%s,MemeticExperimental9,%s,%s,%s\n' % (
    #     config['scenario'] + '_' + str(iteration), str(finish_time), str(services_in_fog), str(services_in_cloud)))

    # Memetic Algorithm with Local Search
    start_time = time.time()  # measure time to complete
    services_placement_count = sg.memeticPlacement(num_creatures, num_generations)
    finish_time = time.time() - start_time

    services_in_fog, services_in_cloud = services_placement_count
    file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    file.write(
        '%s,Memetic,%s,%s,%s\n' % (
            config['scenario'] + '_' + str(iteration), str(finish_time), str(services_in_fog), str(services_in_cloud)))

    file.close()


def run_simulation():
    # logging.config.fileConfig(os.getcwd() + '/logging.ini')
    # simulationDuration = 10000
    simulationDuration = 100000
    algorithms = ['MemeticExperimental',
                  'MemeticExperimental2',
                  'MemeticExperimental6', 'MemeticExperimental7', 'MemeticExperimental8',
                  'Memetic']

    # for the rest of the experiments, run the best one only
    # algorithms = ['MemeticExperimental2', 'MemeticExperimental6', 'MemeticExperimental8', 'Memetic']

    # configs are from ExperimentConfigs file
    config_iterations = []
    for config in configs:
        for iteration in range(config['iterations']):
            config_iterations.append((config, iteration))

    # from itertools import product
    # config_iterations = list(product(configs, range(config['iterations'])))

    fn = partial(run_single_experiment_mp, algorithms=algorithms, simulationDuration=simulationDuration)

    # with Pool(processes=8) as pool:  # for local
    with Pool(processes=128) as pool:  # for AWS  # TODO: change to 128 for AWS
        for _ in pool.imap(fn, config_iterations):
            pass

    # for config in configs:
    #     fn = partial(run_single_experiment, algorithms=algorithms, config=config, simulationDuration=simulationDuration)
    #     # for iteration in range(config['iterations']):
    #     #     fn(iteration)
    #     # with Pool(processes=8) as pool:  # for local
    #     with Pool(processes=128) as pool:  # for AWS
    #         for _ in pool.imap(fn, range(config['iterations'])):
    #             pass
    print("Simulation Done!")


def run_single_experiment_mp(config_iteration, algorithms, simulationDuration):
    config, iteration = config_iteration
    run_single_experiment(iteration, algorithms, config, simulationDuration)


def run_single_experiment(iteration, algorithms, config, simulationDuration):
    # high_traffic = 'high_traffic' # done
    # slow_traffic = 'slow_traffic'  # done

    # weak_fog_devices = 'small_hosts'  # fog devices with below average resources
    # powerful_fog_devices = 'big_hosts'  # fog devices with above average resources

    average = 'average'
    current_setting = average  # third experiment

    folder_results = 'results_' + current_setting + '/current/results_' + config['scenario'] + '_' + str(iteration)
    folder_data = 'data_' + current_setting + '/' + 'data_' + config['scenario'] + '_' + str(iteration)
    os.makedirs(folder_results, exist_ok=True)
    os.makedirs(folder_data, exist_ok=True)
    initialize_experiment(config, iteration, folder_results, folder_data)
    for algorithm in algorithms:
        random.seed(iteration)
        np.random.seed(iteration)
        logging.info("Running experiment type: %s iteration: %i" % (config['scenario'], iteration))

        s_time = time.time()
        main(stop_time=simulationDuration, it=iteration, algorithm=algorithm, config=config,
             folder_results=folder_results,
             folder_data=folder_data)
        print("%s algorithm, %d iteration is done" % (algorithm, iteration))
        print("\n--- %s seconds ---" % (time.time() - s_time))


if __name__ == '__main__':
    run_simulation()
