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
        app_num = user['app']
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

        myAllocation['app'] = app_id
        myAllocation['module_name'] = appDefinition[app_id]['module'][0]['name']
        myAllocation['id_resource'] = resource_id
        myAllocationList.append(myAllocation)

    allAlloc['initialAllocation'] = myAllocationList
    allocationFile = open(folder_results + "/allocDefinitionMemeticExperimental9.json", "w")
    allocationFile.write(json.dumps(allAlloc))
    allocationFile.close()
    print("Memetic experimental 8 initial allocation performed!")
    return (servicesInFog, servicesInCloud)


single_placement(100, 100, 'results_average/current/results_small_0', 'data_average/data_small_0', 'mlmodel.joblib')
