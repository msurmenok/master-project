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
    file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time

    num_creatures = 100
    num_generations = 1000
    mlmodel_filepath = 'mlmodel.joblib'

    # Memetic experimental 9
    start_time = time.time()  # measure time to complete
    services_placement_count = single_placement(num_creatures, num_generations, folder_results, folder_data,
                                                mlmodel_filepath)
    finish_time = time.time() - start_time
    #
    services_in_fog, services_in_cloud = services_placement_count
    file = open(folder_results + "/algorithm_time.csv", 'a+')  # save completion time
    file.write('%s,MemeticExperimental9,%s,%s,%s\n' % (
        config['scenario'] + '_' + str(iteration), str(finish_time), str(services_in_fog), str(services_in_cloud)))

    file.close()


def single_placement(num_creatures, num_generations, folder_results, folder_data, mlmodel_filepath):
    # retrieve al this data from json
    # read netDefinition to get information about available resources
    print('inside single placement')
    netDefinitionFile = open(folder_data + '/netDefinition.json')
    netDefinition = json.load(netDefinitionFile)
    fogDevicesInfo = netDefinition['entity']
    # form netDefinition get all entities that are not gateways

    # read appDefinition to get information about applications
    appDefinitionFile = open(folder_data + '/appDefinition.json')
    appDefinition = json.load(appDefinitionFile)
    # retrieve deadline, module[0]['id or name'], module[0]['CPU'], RAM, STORAGE, PRIORITY

    # to retrieve source ids, and users location we do not use gateways in placement
    usersDefinitionFile = open(folder_data + '/usersDefinition.json')
    usersDefnition = json.load(usersDefinitionFile)
    usersInfo = usersDefnition['sources']

    # populate all these things
    index_to_fogid = {}
    index_to_module_app = {}

    hosts_resources = list()  # numpy array to store host resources and coordinates
    services_requirements = list()  # numpy array to store service requirements, priority, and coordinates
    max_priority = 1
    mapService2App = dict()
    mapServiceId2ServiceName = dict()

    # make a numpy array of fog resources excluding gateway devices
    gatewaysDevices = set()
    cloudId = -1
    distance_to_cloud = 18200

    index = 0
    for user in usersInfo:
        app_num = int(user['app'])
        app_info = appDefinition[app_num]
        module = app_info['module'][0]
        # add mapping id to (app id, module id)
        # Build numpy array: CPU | RAM | STORAGE | PRIORITY | X | Y | deadline
        cpu = module['CPU']
        ram = module['RAM']
        storage = module['STORAGE']
        priority = module['PRIORITY']
        gw_id_src = user['id_resource']
        gatewaysDevices.add(gw_id_src)
        x = fogDevicesInfo[gw_id_src]['x']
        y = fogDevicesInfo[gw_id_src]['y']
        deadline = app_info['deadline']
        services_requirements.append(np.array([cpu, ram, storage, priority, x, y, deadline]))
        index_to_module_app[index] = (app_num, module['id'])
        index += 1
    services_requirements = np.stack(services_requirements)

    index = 0
    for fog_device in fogDevicesInfo:
        if fog_device['type'] == 'CLOUD':
            cloudId = fog_device['id']

        if fog_device['id'] not in gatewaysDevices and fog_device['type'] != 'CLOUD':
            index_to_fogid[index] = fog_device['id']
            index += 1
            # Build numpy array: CPU | MEM | DISK | TIME
            cpu = fog_device['CPU']
            ram = fog_device['RAM']
            storage = fog_device['STORAGE']
            time_availability = fog_device['TIME']
            x = fog_device['x']
            y = fog_device['y']
            ipt = fog_device['IPT']
            hosts_resources.append(np.array([cpu, ram, storage, time_availability, x, y, ipt]))
    hosts_resources = np.stack(hosts_resources)

    # calling Memetic algorithm
    from memetic_experimental9 import memetic_experimental9
    placement = memetic_experimental9(num_creatures, num_generations, services_requirements,
                                      hosts_resources,
                                      max_priority, distance_to_cloud, mlmodel_filepath)
    print("memetic placement: ", placement)

    # convert placement indexes to devices id and save initial placement as json
    servicesInFog = 0
    servicesInCloud = 0
    allAlloc = {}
    myAllocationList = list()

    for i in range(placement.size):
        app_id, module = index_to_module_app[i]
        resource_id = cloudId  # default value if placement is nan
        if np.isnan(placement[i]):
            servicesInCloud += 1
        else:
            resource_id = index_to_fogid[int(placement[i])]
            servicesInFog += 1
        myAllocation = {}

        myAllocation['app'] = str(app_id)
        myAllocation['module_name'] = appDefinition[app_id]['module'][0]['name']
        myAllocation['id_resource'] = resource_id
        myAllocationList.append(myAllocation)

    allAlloc['initialAllocation'] = myAllocationList
    allocationFile = open(folder_data + "/allocDefinitionMemeticExperimental9.json", "w")
    allocationFile.write(json.dumps(allAlloc))
    allocationFile.close()
    print("Memetic experimental 9 initial allocation performed!")
    return (servicesInFog, servicesInCloud)

def run_simulation():
    # logging.config.fileConfig(os.getcwd() + '/logging.ini')
    # simulationDuration = 10000
    simulationDuration = 100000
    # algorithms = ['FirstFitRAM', 'FirstFitTime', 'MemeticWithoutLocalSearch', 'MemeticExperimental',
    #               'MemeticExperimental2', 'MemeticExperimental3', 'MemeticExperimental4', 'MemeticExperimental5',
    #               'MemeticExperimental6', 'MemeticExperimental7', 'MemeticExperimental8',
    #               'Memetic']

    # for the rest of the experiments, run the best one only
    algorithms = ['MemeticExperimental9']

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
    high_traffic = 'high_traffic'  # done
    slow_traffic = 'slow_traffic'  # done

    weak_fog_devices = 'small_hosts'  # fog devices with below average resources
    powerful_fog_devices = 'big_hosts'  # fog devices with above average resources

    # average = 'average2'

    current_setting = weak_fog_devices  # third experiment

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
