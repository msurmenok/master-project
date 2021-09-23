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


def main(stop_time, it, algorithm):
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

    s = Sim(topo, default_results_path=folder_results + "Results_%s_%s_%i_%i" % (algorithm, "small", stop_time, it))

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


def initialize_experiment():
    sg = ExperimentSetup(config=None)
    sg.networkGeneration()
    sg.appGeneration()
    sg.userGeneration()

    # First Fit
    start_time = time.time()  # measure time for placement
    sg.firstFitPlacement()
    finish_time = time.time() - start_time

    file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    file.write('%s, FirstFit, %s\n' % ("small", str(finish_time)))

    # Memetic Algorithm
    start_time = time.time()  # measure time to complete
    sg.memeticPlacement()
    finish_time = time.time() - start_time

    file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    file.write('%s, Memetic, %s\n' % ("small", str(finish_time)))


    file.close()


if __name__ == '__main__':

    # logging.config.fileConfig(os.getcwd() + '/logging.ini')

    nIterations = 10  # iteration for each experiment
    simulationDuration = 10000

    algorithms = ['FirstFit', 'Memetic']

    # Iteration for each experiment changing the seed of randoms
    for iteration in range(nIterations):
        initialize_experiment()
        for algorithm in algorithms:
            random.seed(iteration)
            logging.info("Running experiment it: - %i" % iteration)

            s_time = time.time()
            main(stop_time=simulationDuration, it=iteration, algorithm=algorithm)
            print("%s algorithm, %d iteration is done" % (algorithm, iteration))
            print("\n--- %s seconds ---" % (time.time() - s_time))

    print("Simulation Done!")
