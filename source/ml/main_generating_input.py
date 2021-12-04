from functools import partial
import os
import time
import json
import random
import logging.config
from TrainingExperimentSetup import ExperimentSetup
from TrainingExperimentConfigs import configs
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


def main(stop_time, it, index, algorithm, config, folder_results, folder_data):
    # Create topology from json
    topo = Topology()
    topology_json = json.load(open(folder_data + "/netDefinition.json"))
    # topo.load(topology_json)
    topo.load_all_node_attr(topology_json)
    # topo.write("data_net.gexf")

    # create applications
    data_app = json.load(open(folder_data + "/appDefinition.json"))
    apps = create_applications_from_json(data_app)

    # load placement algorithm
    # allocDefinition_" + algorithm_name + "_" + str(self.iteration) + "_" + str(index) + ".json
    placementJson = json.load(open(folder_data + "/allocDefinition_" + algorithm + "_" + str(it) + "_" + str(
            index) + ".json"))
    placement = JSONPlacement(name="Placement", json=placementJson)

    # load population
    dataPopulation = json.load(open(folder_data + "/usersDefinition.json"))
    pop = JSONPopulation(name="Statical", json=dataPopulation, iteration=it)

    # Routing algorithm
    # selectorPath = MinimunPath()
    selectorPath = DeviceSpeedAwareRouting()
    # folder_results = 'results/'
    s = Sim(topo, default_results_path=folder_results + "Results_%s_%i_%i" % (
        algorithm, it, index))
    # remove link file, we do not use it for ML and it takes a lot of disk space
    os.remove(folder_results + 'Results_' + algorithm + '_' + str(it) + '_' + str(index) + '_link.csv')

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
    # Creating network and users
    sg = ExperimentSetup(config=config, folder_data=folder_data)
    sg.networkGeneration()
    sg.appGeneration()
    sg.userGeneration()

    # Calling placement algorithms
    dataset_size = 100
    num_creatures = 100
    num_generations = 1000

    # sg.memeticExperimentalPlacement7(num_creatures, num_generations, iteration, dataset_size, folder_results)
    sg.memeticExperimentalPlacement8(num_creatures, num_generations, iteration, dataset_size, folder_results)


def run_simulation():
    # logging.config.fileConfig(os.getcwd() + '/logging.ini')
    # simulationDuration = 10000
    simulationDuration = 100000
    # algorithms = ['FirstFitRAM', 'FirstFitTime', 'MemeticWithoutLocalSearch', 'MemeticExperimental',
    #               'MemeticExperimental2', 'MemeticExperimental3', 'MemeticExperimental4', 'MemeticExperimental5',
    #               'MemeticExperimental6', 'MemeticExperimental7', 'MemeticExperimental8', 'MemeticExperimental9',
    #               'Memetic']
    # algorithms = ['MemeticExperimental7', 'MemeticExperimental8']
    algorithms = ['MemeticExperimental8']

    # algorithms = ['FirstFitRAM', 'FirstFitTime']
    # configs are from ExperimentConfigs file
    for config in configs:
        fn = partial(run_single_experiment, algorithms=algorithms, config=config, simulationDuration=simulationDuration)
        # for iteration in range(config['iterations']):
        #     fn(iteration)
        with Pool(processes=128) as pool:
            for _ in pool.imap(fn, range(config['iterations'])):
                pass
    print("Simulation Done!")


def run_single_experiment(iteration, algorithms, config, simulationDuration):
    folder_results = 'results/current/'
    folder_input = 'input/'
    folder_data = 'data/' + 'data_' + config['scenario'] + '_' + str(iteration)
    os.makedirs(folder_results, exist_ok=True)
    os.makedirs(folder_data, exist_ok=True)
    os.makedirs(folder_input, exist_ok=True)
    initialize_experiment(config, iteration, folder_input, folder_data)
    # add one more for loop to run one of the many placements for a single algorithm

    for algorithm in algorithms:
        random.seed(iteration)
        np.random.seed(iteration)
        num_solutions = 100  # may be different size
        for i in range(num_solutions):
            main(stop_time=simulationDuration, it=iteration, index=i, algorithm=algorithm, config=config,
                 folder_results=folder_results,
                 folder_data=folder_data)


if __name__ == '__main__':
    run_simulation()
