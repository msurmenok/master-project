"""
    This is the most simple scenario with a basic topology, some users and a set of apps with only one service.

    @author: Isaac Lera
"""
import os
import time
import json
import random
import logging.config
from ExperimentSetup import ExperimentSetup
from ExperimentConfigs import configs

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

folder_results = "results/"


def main(stop_time, it, algorithm, config):
    # Create topology from json
    topo = Topology()
    topology_json = json.load(open(os.path.dirname(__file__) + "/data/netDefinition.json"))
    # topo.load(topology_json)
    topo.load_all_node_attr(topology_json)
    # topo.write("data_net.gexf")

    # create applications
    data_app = json.load(open(os.path.dirname(__file__) + "/data/appDefinition.json"))
    apps = create_applications_from_json(data_app)

    # load placement algorithm
    placementJson = json.load(open(os.path.dirname(__file__) + "/data/allocDefinition" + algorithm + ".json"))
    placement = JSONPlacement(name="Placement", json=placementJson)

    # load population
    dataPopulation = json.load(open(os.path.dirname(__file__) + "/data/usersDefinition.json"))
    pop = JSONPopulation(name="Statical", json=dataPopulation, iteration=it)

    # Routing algorithm
    # selectorPath = MinimunPath()
    selectorPath = DeviceSpeedAwareRouting()

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


def initialize_experiment(configuration):
    sg = ExperimentSetup(config=configuration)
    sg.networkGeneration()
    sg.appGeneration()
    sg.userGeneration()

    # First Fit RAM
    start_time = time.time()  # measure time for placement
    services_placement_count = sg.firstFitRAMPlacement()
    finish_time = time.time() - start_time

    services_in_fog, services_in_cloud = services_placement_count
    file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    file.write('%s,FirstFitRAM,%s,%s,%s\n' % (config['scenario'], str(finish_time), str(services_in_fog), str(services_in_cloud)))

    # First Fit TIME
    start_time = time.time()  # measure time for placement
    services_placement_count = sg.firstFitTimePlacement()
    finish_time = time.time() - start_time

    services_in_fog, services_in_cloud = services_placement_count
    file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    file.write('%s,FirstFitTime,%s,%s,%s\n' % (config['scenario'], str(finish_time), str(services_in_fog), str(services_in_cloud)))

    # Memetic Algorithm
    num_creatures = 20
    num_generations = 100

    # Memetic Algorithm without Local Search
    start_time = time.time()  # measure time to complete
    services_placement_count = sg.memeticWithoutLocalSearchPlacement(num_creatures, num_generations)
    finish_time = time.time() - start_time

    services_in_fog, services_in_cloud = services_placement_count
    file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    file.write('%s,MemeticWithoutLocalSearch,%s,%s,%s\n' % (config['scenario'], str(finish_time), str(services_in_fog), str(services_in_cloud)))


    # Memetic Algorithm with Local Search
    start_time = time.time()  # measure time to complete
    services_placement_count = sg.memeticPlacement(num_creatures, num_generations)
    finish_time = time.time() - start_time

    services_in_fog, services_in_cloud = services_placement_count
    file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    file.write('%s,Memetic,%s,%s,%s\n' % (config['scenario'], str(finish_time), str(services_in_fog), str(services_in_cloud)))

    file.close()


if __name__ == '__main__':

    # logging.config.fileConfig(os.getcwd() + '/logging.ini')

    # simulationDuration = 10000
    simulationDuration = 100000

    algorithms = ['FirstFitRAM', 'FirstFitTime', 'MemeticWithoutLocalSearch', 'Memetic']
    # algorithms = ['FirstFitRAM', 'FirstFitTime']

    # configs are from ExperimentConfigs file
    for config in configs:
        for iteration in range(config['iterations']):
            initialize_experiment(config)

            for algorithm in algorithms:
                random.seed(iteration)
                logging.info("Running experiment type: %s iteration: %i" % (config['scenario'], iteration))

                s_time = time.time()
                main(stop_time=simulationDuration, it=iteration, algorithm=algorithm, config=config)
                print("%s algorithm, %d iteration is done" % (algorithm, iteration))
                print("\n--- %s seconds ---" % (time.time() - s_time))

    print("Simulation Done!")
