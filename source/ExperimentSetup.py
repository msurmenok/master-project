import math
import sys

import networkx as nx
import operator
import matplotlib.pyplot as plt
import json
import os
import random
import numpy as np
import csv
import copy
from memetic import memetic_algorithm
from memetic_no_local_search import memetic_algorithm_no_local_search


class ExperimentSetup:

    def __init__(self, config):
        self.graphicTerminal = False
        self.verbose_log = False
        self.resultFolder = 'data'

        try:
            os.stat(self.resultFolder)
        except:
            os.mkdir(self.resultFolder)

        # CLOUD
        # CPU | MEM | DISK | TIME
        self.CLOUDSPEED = 10000  # INSTR x MS
        self.CLOUD_PROCESS_RESOURCES = 9999999999999999
        self.CLOUD_RAM_RESOURCES = 9999999999999999  # MB RAM
        self.CLOUD_STORAGE_RESOURCES = 9999999999999999
        self.CLOUD_TIME_AVAILABILITY = 9999999999999999  # cloud always available

        self.CLOUDBW = 125000  # BYTES / MS --> 1000 Mbits/s
        self.CLOUDPR = 500  # MS
        self.CLOUD_X = 18200
        self.CLOUD_Y = 18200
        self.DISTANCE_TO_CLOUD = 25500
        self.cloudId = -1

        # NETWORK
        self.PERCENTAGE_OF_GATEWAYS = 0.25
        self.NUM_FOG_NODES = 50
        self.NUM_GW_DEVICES = round(self.NUM_FOG_NODES * self.PERCENTAGE_OF_GATEWAYS / (
                1 - self.PERCENTAGE_OF_GATEWAYS))  # these devices will not be used for service placement, only as source of events
        self.func_BANDWITDH = lambda: random.randint(75000, 75000)  # BYTES / MS

        # # INTS / MS #random distribution for the speed of the fog devices
        self.func_NODESPEED = lambda: random.randint(500, 1000)
        self.func_NODE_PROCESS_RESOURCES = lambda: round(random.uniform(0.20, 2.00), 2)
        # MB RAM #random distribution for the resources of the fog devices
        self.func_NODE_RAM_RESOURECES = lambda: random.randint(10, 25)
        self.func_NODE_STORAGE_RESOURCES = lambda: random.randint(20, 200)  # MB STORAGE
        self.func_NODE_TIME_AVAILABILITY = lambda: random.randint(100, 2000)  # time availability (in seconds?)

        # Apps and Services
        self.TOTAL_APPS_NUMBER = 10
        # Algorithm for the generation of the random applications
        self.func_APPGENERATION = lambda: nx.gn_graph(random.randint(1, 1))
        # INSTR --> Considering the nodespeed the values should be between 200 & 600 MS
        self.func_SERVICEINSTR = lambda: random.randint(20000, 60000)
        # BYTES --> Considering the BW the values should be between 20 & 60 MS
        self.func_SERVICEMESSAGESIZE = lambda: random.randint(1500000, 4500000)
        # MB of RAM consume by services. Considering noderesources & appgeneration it will be possible to allocate
        # 1 app or +/- 10 services per node
        self.func_SERVICE_RAM_REQUIREMENT = lambda: random.randint(1, 5)
        self.func_SERVICE_PROCESS_REQUIREMENT = lambda: round(random.uniform(0.10, 0.50), 2)
        self.func_SERVICE_STORAGE_REQUIREMENT = lambda: random.randint(10, 50)
        self.MAX_PRIORITY = 1
        self.func_SERVICE_PRIORITY = lambda: random.randint(0, self.MAX_PRIORITY)
        # TODO: change back after the experiment
        self.func_APPDEADLINE = lambda: random.randint(5000000, 5000000)  # MS
        # self.func_APPDEADLINE = lambda: random.randint(300, 500000)  # MS

        # Users and IoT devices
        # App's popularity. This value define the probability of source request an application
        self.func_REQUESTPROB = lambda: random.random() / 4
        self.func_USERREQRAT = lambda: random.randint(200, 1000)  # MS

        self.myDeadlines = [487203.22, 487203.22, 487203.22, 474.51, 302.05, 831.04, 793.26, 1582.21, 2214.64,
                            374046.40, 420476.14, 2464.69, 97999.14, 2159.73, 915.16, 1659.97, 1059.97, 322898.56,
                            1817.51, 406034.73, 487203.22, 487203.22, 487203.22, 474.51, 302.05, 831.04, 793.26,
                            1582.21, 2214.64,
                            374046.40]

        # random.seed(15612357)
        self.FGraph = None
        if config:
            self.loadConfigurations(config)

    def loadConfigurations(self, config):
        scenario = config['scenario']
        if scenario == 'tiny':
            self.NUM_FOG_NODES = 20
            self.TOTAL_APPS_NUMBER = 5
            self.NUM_GW_DEVICES = round(
                self.NUM_FOG_NODES * self.PERCENTAGE_OF_GATEWAYS / (1 - self.PERCENTAGE_OF_GATEWAYS))
        if scenario == 'small':
            self.NUM_FOG_NODES = 50
            self.TOTAL_APPS_NUMBER = 10
            self.NUM_GW_DEVICES = round(
                self.NUM_FOG_NODES * self.PERCENTAGE_OF_GATEWAYS / (1 - self.PERCENTAGE_OF_GATEWAYS))
        if scenario == 'medium':
            self.NUM_FOG_NODES = 50
            self.TOTAL_APPS_NUMBER = 20
            self.NUM_GW_DEVICES = round(
                self.NUM_FOG_NODES * self.PERCENTAGE_OF_GATEWAYS / (1 - self.PERCENTAGE_OF_GATEWAYS))
        if scenario == 'large':
            self.NUM_FOG_NODES = 100
            self.TOTAL_APPS_NUMBER = 40
            self.NUM_GW_DEVICES = round(
                self.NUM_FOG_NODES * self.PERCENTAGE_OF_GATEWAYS / (1 - self.PERCENTAGE_OF_GATEWAYS))
        if scenario == 'verylarge':
            self.NUM_FOG_NODES = 150
            self.TOTAL_APPS_NUMBER = 100
            self.NUM_GW_DEVICES = round(
                self.NUM_FOG_NODES * self.PERCENTAGE_OF_GATEWAYS / (1 - self.PERCENTAGE_OF_GATEWAYS))

    def networkGeneration(self):
        # Generating network topology
        # Using barbasi algorithm for the generation of the network topology
        self.G = nx.barabasi_albert_graph(n=(self.NUM_GW_DEVICES + self.NUM_FOG_NODES), m=2)
        self.devices = list()

        self.nodeResources = {}
        self.nodeFreeResources = {}
        self.nodeSpeed = {}

        node_positions = nx.spring_layout(self.G, seed=15612357, scale=500, center=[500, 500])

        for i in self.G.nodes:
            # CPU | MEM | DISK | TIME
            node_cpu = self.func_NODE_PROCESS_RESOURCES()
            node_ram = self.func_NODE_RAM_RESOURECES()
            node_storage = self.func_NODE_STORAGE_RESOURCES()
            node_time_availability = self.func_NODE_TIME_AVAILABILITY()

            current_node_resources = {}
            current_node_resources['CPU'] = node_cpu
            current_node_resources['RAM'] = node_ram
            current_node_resources['STORAGE'] = node_storage
            current_node_resources['TIME'] = node_time_availability
            current_node_resources['x'] = round(node_positions[i][0])
            current_node_resources['y'] = round(node_positions[i][1])
            current_node_resources['TYPE'] = 'FOG'

            # replaced single RAM value with the dictionary of multiple resource requirements
            # self.nodeResources[i] = eval(self.func_NODE_RAM_RESOURECES)
            self.nodeResources[i] = current_node_resources
            self.nodeSpeed[i] = self.func_NODESPEED()

            myNode = {}
            myNode['id'] = i
            myNode['RAM'] = self.nodeResources[i]['RAM']
            myNode['FRAM'] = self.nodeResources[i]['RAM']
            myNode['IPT'] = self.nodeSpeed[i]
            myNode['CPU'] = self.nodeResources[i]['CPU']
            myNode['STORAGE'] = self.nodeResources[i]['STORAGE']
            myNode['TIME'] = self.nodeResources[i]['TIME']
            myNode['x'] = self.nodeResources[i]['x']
            myNode['y'] = self.nodeResources[i]['y']
            myNode['type'] = self.nodeResources[i]['TYPE']
            self.devices.append(myNode)

        for e in self.G.edges:
            # get 2 nodes to calculate PR between them based on their location
            node1 = next((x for x in self.devices if x['id'] == e[0]), None)
            node2 = next((x for x in self.devices if x['id'] == e[1]), None)
            scale = 50  # scale the result so that maximum transmission b/w nodes is 10 ms

            minimum_pr = 2  # if calculated value is below 2, set it to 2
            distance = math.sqrt((node1['x'] - node2['x']) ** 2 + (node1['y'] - node2['y']) ** 2)
            pr = max(int(distance / scale), minimum_pr)

            self.G[e[0]][e[1]]['PR'] = pr
            self.G[e[0]][e[1]]['BW'] = self.func_BANDWITDH()

        myEdges = list()
        for e in self.G.edges:
            myLink = {}
            myLink['s'] = e[0]
            myLink['d'] = e[1]
            myLink['PR'] = self.G[e[0]][e[1]]['PR']
            myLink['BW'] = self.G[e[0]][e[1]]['BW']

            myEdges.append(myLink)

        # Find edge gateway devices (lowest betweenness centrality) and cloud gateway device (highest betweenness)
        centralityValuesNoOrdered = nx.betweenness_centrality(self.G, weight="weight")
        centralityValues = sorted(centralityValuesNoOrdered.items(), key=operator.itemgetter(1), reverse=True)

        self.gatewaysDevices = set()
        self.fogDevices = set()
        self.cloudgatewaysDevices = set()

        highestCentrality = centralityValues[0][1]

        for device in centralityValues:
            if device[1] == highestCentrality:
                self.cloudgatewaysDevices.add(device[0])

        # initialIndx = int((1 - self.PERCENTATGEOFGATEWAYS) * len(self.G.nodes))  # Getting the indexes for the GWs nodes
        first_gw_index = len(self.G.nodes) - self.NUM_GW_DEVICES
        for idDev in range(len(self.G.nodes)):
            if idDev < first_gw_index:
                self.fogDevices.add(centralityValues[idDev][0])
            else:
                self.gatewaysDevices.add(centralityValues[idDev][0])

        # create cloud node
        self.cloudId = len(self.G.nodes)
        myNode = {}
        myNode['id'] = self.cloudId
        myNode['RAM'] = self.CLOUD_RAM_RESOURCES
        myNode['FRAM'] = self.CLOUD_RAM_RESOURCES
        myNode['IPT'] = self.CLOUDSPEED
        myNode['type'] = 'CLOUD'
        myNode['CPU'] = self.CLOUD_PROCESS_RESOURCES
        myNode['STORAGE'] = self.CLOUD_STORAGE_RESOURCES
        myNode['TIME'] = self.CLOUD_TIME_AVAILABILITY
        myNode['x'] = self.CLOUD_X
        myNode['y'] = self.CLOUD_Y
        self.devices.append(myNode)
        self.fogDevices.add(self.cloudId)

        # Adding Cloud's resource to self.nodeResources
        current_node_resources = {}
        current_node_resources['CPU'] = self.CLOUD_PROCESS_RESOURCES
        current_node_resources['RAM'] = self.CLOUD_RAM_RESOURCES
        current_node_resources['STORAGE'] = self.CLOUD_STORAGE_RESOURCES
        current_node_resources['TIME'] = self.CLOUD_TIME_AVAILABILITY
        current_node_resources['x'] = self.CLOUD_X
        current_node_resources['y'] = self.CLOUD_Y
        current_node_resources['TYPE'] = 'CLOUD'
        self.nodeResources[self.cloudId] = current_node_resources

        # At the begging all the resources on the nodes are free
        self.nodeFreeResources = copy.deepcopy(self.nodeResources)

        # add edge between cloud gateway and cloud node
        for cloudGtw in self.cloudgatewaysDevices:
            myLink = {}
            myLink['s'] = cloudGtw
            myLink['d'] = self.cloudId
            myLink['PR'] = self.CLOUDPR
            myLink['BW'] = self.CLOUDBW
            myEdges.append(myLink)

        # Plotting the graph with all the element
        if self.graphicTerminal:
            self.FGraph = self.G
            self.FGraph.add_node(self.cloudId)
            for gw_node in list(self.cloudgatewaysDevices):
                self.FGraph.add_edge(gw_node, self.cloudId, PR=self.CLOUDPR, BW=self.CLOUDBW)
            fig, ax = plt.subplots()
            pos = nx.spring_layout(self.FGraph, pos=node_positions,
                                   fixed=list(range(0, self.NUM_FOG_NODES + self.NUM_GW_DEVICES)), seed=15612357,
                                   scale=500,
                                   center=[500, 500])
            nx.draw(self.FGraph, pos)
            nx.draw_networkx_labels(self.FGraph, pos, font_size=8)
            plt.show()
            # Unix
            fig.savefig(os.path.dirname(__file__) + '/' + self.resultFolder + '/plots/netTopology.png')
            plt.close(fig)  # close the figure

        # export to JSON
        netJson = {}
        netJson['entity'] = self.devices
        netJson['link'] = myEdges

        # Unix
        netFile = open(os.path.dirname(__file__) + '/' + self.resultFolder + "/netDefinition.json", "w")
        netFile.write(json.dumps(netJson))
        netFile.close()

    def appGeneration(self):
        # Apps generation

        self.numberOfServices = 0
        self.apps = list()
        self.appsDeadlines = {}
        self.appsResources = list()
        self.appsSourceService = list()
        self.appsSourceMessage = list()
        self.appsTotalMIPS = list()
        self.appsTotalServices = list()
        self.mapService2App = list()
        self.mapServiceId2ServiceName = list()

        appJson = list()
        self.servicesResources = {}

        for i in range(0, self.TOTAL_APPS_NUMBER):
            myApp = {}
            APP = self.func_APPGENERATION()

            mylabels = {}

            for n in range(0, len(APP.nodes)):
                mylabels[n] = str(n)

            edgeList_ = list()

            # Reverting the direction of the edges from Source to Modules

            for m in APP.edges:
                edgeList_.append(m)
            for m in edgeList_:
                APP.remove_edge(m[0], m[1])
                APP.add_edge(m[1], m[0])

            if self.graphicTerminal:
                fig, ax = plt.subplots()
                pos = nx.spring_layout(APP, seed=15612357)
                nx.draw(APP, pos, labels=mylabels, font_size=8)
                # Unix
                fig.savefig(os.path.dirname(__file__) + '/' + self.resultFolder + '/plots/app_%s.png' % i)
                plt.close(fig)  # close the figure
                plt.show()

            mapping = dict(zip(APP.nodes(), range(self.numberOfServices, self.numberOfServices + len(APP.nodes))))
            APP = nx.relabel_nodes(APP, mapping)

            self.numberOfServices = self.numberOfServices + len(APP.nodes)
            self.apps.append(APP)
            for j in APP.nodes:
                my_resource_requirements = {}
                my_resource_requirements['CPU'] = self.func_SERVICE_PROCESS_REQUIREMENT()
                my_resource_requirements['RAM'] = self.func_SERVICE_RAM_REQUIREMENT()
                my_resource_requirements['STORAGE'] = self.func_SERVICE_STORAGE_REQUIREMENT()
                my_resource_requirements['PRIORITY'] = self.func_SERVICE_PRIORITY()
                self.servicesResources[j] = my_resource_requirements

            self.appsResources.append(self.servicesResources)

            topologicorder_ = list(nx.topological_sort(APP))
            source = topologicorder_[0]

            self.appsSourceService.append(source)

            self.appsDeadlines[i] = self.func_APPDEADLINE()

            myApp['id'] = i
            myApp['name'] = str(i)
            myApp['deadline'] = self.appsDeadlines[i]

            myApp['module'] = list()

            edgeNumber = 0
            myApp['message'] = list()

            myApp['transmission'] = list()

            totalMIPS = 0

            for n in APP.nodes:
                self.mapService2App.append(str(i))
                self.mapServiceId2ServiceName.append(str(i) + '_' + str(n))
                myNode = {}
                myNode['id'] = n
                myNode['name'] = str(i) + '_' + str(n)
                myNode['CPU'] = self.servicesResources[n]['CPU']
                myNode['RAM'] = self.servicesResources[n]['RAM']
                myNode['STORAGE'] = self.servicesResources[n]['STORAGE']
                myNode['PRIORITY'] = self.servicesResources[n]['PRIORITY']
                myNode['type'] = 'MODULE'
                if source == n:
                    myEdge = {}
                    myEdge['id'] = edgeNumber
                    edgeNumber = edgeNumber + 1
                    myEdge['name'] = "M.USER.APP." + str(i)
                    myEdge['s'] = "None"
                    myEdge['d'] = str(i) + '_' + str(n)
                    myEdge['instructions'] = self.func_SERVICEINSTR()
                    totalMIPS = totalMIPS + myEdge['instructions']
                    myEdge['bytes'] = self.func_SERVICEMESSAGESIZE()
                    myApp['message'].append(myEdge)
                    self.appsSourceMessage.append(myEdge)

                    for o in APP.edges:
                        if o[0] == source:
                            myTransmission = {}
                            myTransmission['module'] = str(i) + '_' + str(source)
                            myTransmission['message_in'] = "M.USER.APP." + str(i)
                            myTransmission['message_out'] = str(i) + '_(' + str(o[0]) + "-" + str(o[1]) + ")"
                            myApp['transmission'].append(myTransmission)
                    # in case application has just one module, add transmission from sensor to that single module
                    if not APP.edges:
                        myTransmission = {}
                        myTransmission['module'] = str(i) + '_' + str(source)
                        myTransmission['message_in'] = "M.USER.APP." + str(i)
                        myApp['transmission'].append(myTransmission)

                myApp['module'].append(myNode)

            for n in APP.edges:
                myEdge = {}
                myEdge['id'] = edgeNumber
                edgeNumber = edgeNumber + 1
                myEdge['name'] = str(i) + '_(' + str(n[0]) + "-" + str(n[1]) + ")"
                myEdge['s'] = str(i) + '_' + str(n[0])
                myEdge['d'] = str(i) + '_' + str(n[1])
                myEdge['instructions'] = self.func_SERVICEINSTR()
                totalMIPS = totalMIPS + myEdge['instructions']
                myEdge['bytes'] = self.func_SERVICEMESSAGESIZE()
                myApp['message'].append(myEdge)
                destNode = n[1]
                for o in APP.edges:
                    if o[0] == destNode:
                        myTransmission = {}
                        myTransmission['module'] = str(i) + '_' + str(n[1])
                        myTransmission['message_in'] = str(i) + '_(' + str(n[0]) + "-" + str(n[1]) + ")"
                        myTransmission['message_out'] = str(i) + '_(' + str(o[0]) + "-" + str(o[1]) + ")"
                        myApp['transmission'].append(myTransmission)

            for n in APP.nodes:
                outgoingEdges = False
                for m in APP.edges:
                    if m[0] == n:
                        outgoingEdges = True
                        break
                if not outgoingEdges:
                    for m in APP.edges:
                        if m[1] == n:
                            myTransmission = {}
                            myTransmission['module'] = str(i) + '_' + str(n)
                            myTransmission['message_in'] = str(i) + '_(' + str(m[0]) + "-" + str(m[1]) + ")"
                            myApp['transmission'].append(myTransmission)

            self.appsTotalMIPS.append(totalMIPS)
            self.appsTotalServices.append(len(APP.nodes()))

            appJson.append(myApp)

        appFile = open(self.resultFolder + "/appDefinition.json", "w")
        appFile.write(json.dumps(appJson))
        appFile.close()

    def userGeneration(self):
        # Generation of the IoT devices (users)
        userJson = {}

        self.myUsers = list()

        self.appsRequests = list()
        for i in range(0, self.TOTAL_APPS_NUMBER):
            userRequestList = set()
            probOfRequested = self.func_REQUESTPROB()
            # probOfRequested = -1
            atLeastOneAllocated = False
            for j in self.gatewaysDevices:
                rand = random.random()
                if rand < probOfRequested:
                    myOneUser = {}
                    myOneUser['app'] = str(i)
                    myOneUser['message'] = "M.USER.APP." + str(i)
                    myOneUser['id_resource'] = j
                    myOneUser['lambda'] = self.func_USERREQRAT()
                    userRequestList.add(j)
                    self.myUsers.append(myOneUser)
                    atLeastOneAllocated = True
            if not atLeastOneAllocated:
                j = random.randint(0, len(self.gatewaysDevices) - 1)
                myOneUser = {}
                myOneUser['app'] = str(i)
                myOneUser['message'] = "M.USER.APP." + str(i)
                myOneUser['id_resource'] = list(self.gatewaysDevices)[j]  # Random GW to host the request
                myOneUser['lambda'] = self.func_USERREQRAT()
                userRequestList.add(list(self.gatewaysDevices)[j])
                self.myUsers.append(myOneUser)
            self.appsRequests.append(userRequestList)

        userJson['sources'] = self.myUsers

        userFile = open(self.resultFolder + "/usersDefinition.json", "w")
        userFile.write(json.dumps(userJson))
        userFile.close()

    def memeticWithoutLocalSearchPlacement(self, num_creatures, num_generations):
        index_to_fogid = {}
        index_to_module_app = {}

        hosts_resources = list()  # numpy array to store host resources and coordinates
        services_requirements = list()  # numpy array to store service requirements, priority, and coordinates

        # make a numpy array of fog resources excluding gateway devices
        # map index to fog device id
        index = 0

        for k, v in self.nodeResources.items():
            if k not in self.gatewaysDevices and k is not self.cloudId:
                index_to_fogid[index] = k
                index += 1
                # Build numpy array: CPU | MEM | DISK | TIME
                cpu = v['CPU']
                ram = v['RAM']
                storage = v['STORAGE']
                time_availability = v['TIME']
                x = v['x']
                y = v['y']
                hosts_resources.append(np.array([cpu, ram, storage, time_availability, x, y]))
        hosts_resources = np.stack(hosts_resources)

        index = 0
        for app_num, app in enumerate(self.appsRequests):
            for instance, gw_id in enumerate(self.appsRequests[app_num]):
                for module in list(self.apps[app_num].nodes):
                    # add mapping id to (app id, module id)
                    index_to_module_app[index] = (app_num, module)
                    index += 1
                    # Build numpy array: CPU | RAM | STORAGE | PRIORITY | X | Y
                    res_required = self.servicesResources[module]
                    cpu = res_required['CPU']
                    ram = res_required['RAM']
                    storage = res_required['STORAGE']
                    priority = res_required['PRIORITY']
                    x = self.nodeResources[gw_id]['x']
                    y = self.nodeResources[gw_id]['y']
                    services_requirements.append(np.array([cpu, ram, storage, priority, x, y]))
        services_requirements = np.stack(services_requirements)

        # calling Memetic algorithm

        placement = memetic_algorithm_no_local_search(num_creatures, num_generations, services_requirements,
                                                      hosts_resources,
                                                      self.MAX_PRIORITY, self.DISTANCE_TO_CLOUD)
        print("memetic placement: ", placement)

        # convert placement indexes to devices id and save initial placement as json
        servicesInFog = 0
        servicesInCloud = 0
        allAlloc = {}
        myAllocationList = list()

        for i in range(placement.size):
            app_id, module = index_to_module_app[i]
            resource_id = self.cloudId  # default value if placement is nan
            if np.isnan(placement[i]):
                servicesInCloud += 1
            else:
                resource_id = index_to_fogid[int(placement[i])]
                servicesInFog += 1
            myAllocation = {}
            myAllocation['app'] = self.mapService2App[module]
            myAllocation['module_name'] = self.mapServiceId2ServiceName[module]
            myAllocation['id_resource'] = resource_id
            myAllocationList.append(myAllocation)

        allAlloc['initialAllocation'] = myAllocationList
        allocationFile = open(self.resultFolder + "/allocDefinitionMemeticWithoutLocalSearch.json", "w")
        allocationFile.write(json.dumps(allAlloc))
        allocationFile.close()
        print("Memetic without Local Search initial allocation performed!")
        return (servicesInFog, servicesInCloud)

    def memeticPlacement(self, num_creatures, num_generations):
        index_to_fogid = {}
        index_to_module_app = {}

        hosts_resources = list()  # numpy array to store host resources and coordinates
        services_requirements = list()  # numpy array to store service requirements, priority, and coordinates

        # make a numpy array of fog resources excluding gateway devices
        # map index to fog device id
        index = 0

        for k, v in self.nodeResources.items():
            if k not in self.gatewaysDevices and k is not self.cloudId:
                index_to_fogid[index] = k
                index += 1
                # Build numpy array: CPU | MEM | DISK | TIME
                cpu = v['CPU']
                ram = v['RAM']
                storage = v['STORAGE']
                time_availability = v['TIME']
                x = v['x']
                y = v['y']
                hosts_resources.append(np.array([cpu, ram, storage, time_availability, x, y]))
        hosts_resources = np.stack(hosts_resources)

        index = 0
        for app_num, app in enumerate(self.appsRequests):
            for instance, gw_id in enumerate(self.appsRequests[app_num]):
                for module in list(self.apps[app_num].nodes):
                    # add mapping id to (app id, module id)
                    index_to_module_app[index] = (app_num, module)
                    index += 1
                    # Build numpy array: CPU | RAM | STORAGE | PRIORITY | X | Y
                    res_required = self.servicesResources[module]
                    cpu = res_required['CPU']
                    ram = res_required['RAM']
                    storage = res_required['STORAGE']
                    priority = res_required['PRIORITY']
                    x = self.nodeResources[gw_id]['x']
                    y = self.nodeResources[gw_id]['y']
                    services_requirements.append(np.array([cpu, ram, storage, priority, x, y]))
        services_requirements = np.stack(services_requirements)

        # calling Memetic algorithm
        placement = memetic_algorithm(num_creatures, num_generations, services_requirements, hosts_resources,
                                      self.MAX_PRIORITY, self.DISTANCE_TO_CLOUD)
        print("memetic placement: ", placement)

        # convert placement indexes to devices id and save initial placement as json
        servicesInFog = 0
        servicesInCloud = 0
        allAlloc = {}
        myAllocationList = list()

        for i in range(placement.size):
            app_id, module = index_to_module_app[i]
            resource_id = self.cloudId  # default value if placement is nan
            if np.isnan(placement[i]):
                servicesInCloud += 1
            else:
                resource_id = index_to_fogid[int(placement[i])]
                servicesInFog += 1
            myAllocation = {}
            myAllocation['app'] = self.mapService2App[module]
            myAllocation['module_name'] = self.mapServiceId2ServiceName[module]
            myAllocation['id_resource'] = resource_id
            myAllocationList.append(myAllocation)

        allAlloc['initialAllocation'] = myAllocationList
        allocationFile = open(self.resultFolder + "/allocDefinitionMemetic.json", "w")
        allocationFile.write(json.dumps(allAlloc))
        allocationFile.close()
        print("Memetic initial allocation performed!")
        return (servicesInFog, servicesInCloud)

    def firstFitRAMPlacement(self):
        # self.nodeFreeResources = copy.deepcopy(self.nodeResources)
        servicesInFog = 0
        servicesInCloud = 0
        allAlloc = {}
        myAllocationList = list()
        # random.seed(datetime.now())

        aux = copy.deepcopy(self.nodeResources)
        aux = sorted(aux.items(), key=lambda x: x[1]['RAM'])
        # aux = sorted(self.nodeResources.items(), key=lambda x: x[1]['TIME'], reverse=True)

        sorted_nodeResources = [list(sub_list) for sub_list in aux]

        for app_num, app in zip(range(0, len(self.appsRequests)), self.appsRequests):
            for instance in range(0, len(self.appsRequests[app_num])):
                for module in list(self.apps[app_num].nodes):
                    flag = True
                    iterations = 0
                    while flag and iterations < len(sorted_nodeResources):
                        # Chosing the node with less resources to host the service
                        index = iterations
                        iterations += 1
                        if sorted_nodeResources[index][0] in self.gatewaysDevices:
                            continue
                        # Checking if the node has resource to host the service
                        res_required = self.servicesResources[module]
                        if res_required['CPU'] <= sorted_nodeResources[index][1]['CPU'] and res_required['STORAGE'] <= \
                                sorted_nodeResources[index][1]['STORAGE'] and res_required['RAM'] <= \
                                sorted_nodeResources[index][1]['RAM']:
                            remaining_resources = sorted_nodeResources[index][1]
                            remaining_resources['CPU'] -= res_required['CPU']
                            remaining_resources['STORAGE'] -= res_required['STORAGE']
                            remaining_resources['RAM'] -= res_required['RAM']

                            # Updating sorted resource list
                            sorted_nodeResources[index][1] = remaining_resources
                            # Updating nodeFreeResources
                            self.nodeFreeResources[sorted_nodeResources[index][0]] = remaining_resources
                            myAllocation = {}
                            myAllocation['app'] = self.mapService2App[module]
                            myAllocation['module_name'] = self.mapServiceId2ServiceName[module]
                            myAllocation['id_resource'] = sorted_nodeResources[index][0]
                            flag = False
                            myAllocationList.append(myAllocation)
                            if sorted_nodeResources[index][0] != self.cloudId:
                                servicesInFog += 1
                            else:
                                servicesInCloud += 1
                    if flag and iterations == (len(sorted_nodeResources) - 1):
                        print(
                            "After %i iterations it was not possible to place the module %i using the FirstFitRAMPlacement" \
                            % (iterations, module))
                        exit()
        allAlloc['initialAllocation'] = myAllocationList

        allocationFile = open(self.resultFolder + "/allocDefinitionFirstFitRAM.json", "w")
        allocationFile.write(json.dumps(allAlloc))
        allocationFile.close()

        # Keeping nodes' resources
        final_nodeResources = sorted(self.nodeResources.items(), key=operator.itemgetter(0))

        print("FirstFitRAM initial allocation performed!")
        return (servicesInFog, servicesInCloud)

    def firstFitTimePlacement(self):
        self.nodeFreeResources = copy.deepcopy(self.nodeResources)
        servicesInFog = 0
        servicesInCloud = 0
        allAlloc = {}
        myAllocationList = list()
        # random.seed(datetime.now())

        aux = copy.deepcopy(self.nodeResources)
        aux = sorted(aux.items(), key=lambda x: x[1]['TIME'], reverse=True)
        aux.append(aux.pop(0))

        # we do not want cloud as the first option for placement, move cloud id to the end
        sorted_nodeResources = [list(sub_list) for sub_list in aux]

        for app_num, app in zip(range(0, len(self.appsRequests)), self.appsRequests):
            for instance in range(0, len(self.appsRequests[app_num])):
                for module in list(self.apps[app_num].nodes):
                    flag = True
                    iterations = 0
                    while flag and iterations < len(sorted_nodeResources):
                        # Chosing the node with less resources to host the service
                        index = iterations
                        iterations += 1
                        if sorted_nodeResources[index][0] in self.gatewaysDevices:
                            continue
                        # Checking if the node has resource to host the service
                        res_required = self.servicesResources[module]

                        if res_required['CPU'] <= sorted_nodeResources[index][1]['CPU'] and res_required['STORAGE'] <= \
                                sorted_nodeResources[index][1]['STORAGE'] and res_required['RAM'] <= \
                                sorted_nodeResources[index][1]['RAM']:
                            remaining_resources = sorted_nodeResources[index][1]
                            remaining_resources['CPU'] -= res_required['CPU']
                            remaining_resources['STORAGE'] -= res_required['STORAGE']
                            remaining_resources['RAM'] -= res_required['RAM']

                            # Updating sorted resource list
                            sorted_nodeResources[index][1] = remaining_resources
                            # Updating nodeFreeResources
                            self.nodeFreeResources[sorted_nodeResources[index][0]] = remaining_resources
                            myAllocation = {}
                            myAllocation['app'] = self.mapService2App[module]
                            myAllocation['module_name'] = self.mapServiceId2ServiceName[module]
                            myAllocation['id_resource'] = sorted_nodeResources[index][0]
                            flag = False
                            myAllocationList.append(myAllocation)
                            if sorted_nodeResources[index][0] != self.cloudId:
                                servicesInFog += 1
                            else:
                                servicesInCloud += 1
                    if flag and iterations == (len(sorted_nodeResources) - 1):
                        print(
                            "After %i iterations it was not possible to place the module %i using the FirstFitTimelacement" \
                            % (iterations, module))
                        exit()
        allAlloc['initialAllocation'] = myAllocationList

        allocationFile = open(self.resultFolder + "/allocDefinitionFirstFitTime.json", "w")
        allocationFile.write(json.dumps(allAlloc))
        allocationFile.close()

        # Keeping nodes' resources
        final_nodeResources = sorted(self.nodeResources.items(), key=operator.itemgetter(0))

        print("FirstFitTime initial allocation performed!")
        return ((servicesInFog, servicesInCloud))
