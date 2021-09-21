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


class ExperimentSetup:

    def __init__(self, config):
        self.graphicTerminal = True
        self.verbose_log = False
        self.resultFolder = 'conf'

        try:
            os.stat(self.resultFolder)
        except:
            os.mkdir(self.resultFolder)

        # CLOUD
        # CPU | MEM | DISK | TIME
        self.CLOUDSPEED = 10000  # INSTR x MS
        self.CLOUD_PROCESS_RESOURCES = 999
        self.CLOUD_RAM_RESOURCES = 9999999999999999  # MB RAM
        self.CLOUD_STORAGE_RESOURCES = 99999999999999999
        self.CLOUD_TIME_AVAILABILITY = sys.maxsize  # cloud always available

        self.CLOUDBW = 125000  # BYTES / MS --> 1000 Mbits/s
        self.CLOUDPR = 500  # MS
        self.CLOUD_X = 18200
        self.CLOUD_Y = 18200

        # NETWORK
        # self.PERCENTATGEOFGATEWAYS = 0.25
        self.NUM_GW_DEVICES = 15  # these devices will not be used for service placement, only as source of events
        self.NUM_FOG_NODES = 50
        self.func_PROPAGATIONTIME = "random.randint(2,10)"  # MS
        self.func_BANDWITDH = "random.randint(75000,75000)"  # BYTES / MS

        # CPU | MEM | DISK | TIME
        self.func_NODESPEED = "random.randint(500,1000)"  # INTS / MS #random distribution for the speed of the fog devices
        self.func_NODE_PROCESS_RESOURCES = "round(random.uniform(0.20, 1.00),2)"
        self.func_NODE_RAM_RESOURECES = "random.randint(10,25)"  # MB RAM #random distribution for the resources of the fog devices
        self.func_NODE_STORAGE_RESOURCES = "round(random.uniform(0.20, 1.00),2)"  # MB STORAGE
        self.func_NODE_TIME_AVAILABILITY = "random.randint(100, 2000)"  # time availability (in seconds?)

        # Apps and Services
        self.TOTAL_APPS_NUMBER = 10
        self.func_APPGENERATION = "nx.gn_graph(random.randint(1,1))"  # Algorithm for the generation of the random applications
        self.func_SERVICEINSTR = "random.randint(20000,60000)"  # INSTR --> Considering the nodespeed the values should be between 200 & 600 MS
        self.func_SERVICEMESSAGESIZE = "random.randint(1500000,4500000)"  # BYTES --> Considering the BW the values should be between 20 & 60 MS
        # CPU | MEM | DISK | PRIORITY
        self.func_SERVICE_PROCESS_REQUIREMENT = "round(random.uniform(0.20, 0.50),2)"
        self.func_SERVICE_RAM_REQUIREMENT = "random.randint(1,5)"  # MB of RAM consume by services. Considering noderesources & appgeneration it will be possible to allocate 1 app or +/- 10 services per node
        self.func_SERVICE_STORAGE_REQUIREMENT = "round(random.uniform(0.20, 0.80),2)"
        self.func_SERVICE_PRIORITY = "random.randint(0,1)"
        # self.func_APPDEADLINE = "random.randint(2600,6600)"  # MS

        # Users and IoT devices
        self.func_REQUESTPROB = "random.random()/4"  # App's popularity. This value define the probability of source request an application
        self.func_USERREQRAT = "random.randint(200,1000)"  # MS

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
        pass

    def networkGeneration(self):
        # Generating network topology
        # Using barbasi algorithm for the generation of the network topology
        self.G = nx.barabasi_albert_graph(n=(self.NUM_GW_DEVICES + self.NUM_FOG_NODES), m=2)
        self.devices = list()

        self.nodeResources = {}
        self.nodeFreeResources = {}
        self.nodeSpeed = {}

        for i in self.G.nodes:
            '''
                    self.func_NODE_PROCESS_RESOURCES = "random.uniform(0.20, 1.00)"
        self.func_NODE_RAM_RESOURECES = "random.randint(10,25)"  # MB RAM #random distribution for the resources of the fog devices
        self.func_NODE_STORAGE_RESOURCES = "random.uniform(0.20, 1.00)"  # MB STORAGE
        self.func_NODE_TIME_AVAILABILITY = "random.randint(100, 2000)" 
            '''
            # CPU | MEM | DISK | TIME
            node_cpu = eval(self.func_NODE_PROCESS_RESOURCES)
            node_ram = eval(self.func_NODE_RAM_RESOURECES)
            node_storage = eval(self.func_NODE_STORAGE_RESOURCES)
            node_time_availability = eval(self.func_NODE_TIME_AVAILABILITY)

            current_node_resources = {}
            current_node_resources['CPU'] = node_cpu
            current_node_resources['RAM'] = node_ram
            current_node_resources['STORAGE'] = node_storage
            current_node_resources['TIME'] = node_time_availability

            # replaced single RAM value with the dictionary of multiple resource requirements
            # self.nodeResources[i] = eval(self.func_NODE_RAM_RESOURECES)
            self.nodeResources[i] = current_node_resources
            self.nodeSpeed[i] = eval(self.func_NODESPEED)

        node_positions = nx.spring_layout(self.G, seed=15612357, scale=500, center=[500, 500])

        for i in self.G.nodes:
            myNode = {}
            myNode['id'] = i
            myNode['RAM'] = self.nodeResources[i]['RAM']
            myNode['FRAM'] = self.nodeResources[i]['RAM']
            myNode['IPT'] = self.nodeSpeed[i]
            myNode['CPU'] = self.nodeResources[i]['CPU']
            myNode['STORAGE'] = self.nodeResources[i]['STORAGE']
            myNode['TIME'] = self.nodeResources[i]['TIME']
            myNode['x'] = round(node_positions[i][0])
            myNode['y'] = round(node_positions[i][1])
            myNode['type'] = 'FOG'
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
            self.G[e[0]][e[1]]['BW'] = eval(self.func_BANDWITDH)

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
        self.nodeResources[self.cloudId] = current_node_resources

        # At the begging all the resources on the nodes are free
        self.nodeFreeResources = self.nodeResources

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
            pos = nx.spring_layout(self.FGraph, pos=node_positions, fixed=list(range(0, 50)), seed=15612357, scale=500,
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
            APP = eval(self.func_APPGENERATION)

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
                my_resource_requirements['CPU'] = eval(self.func_SERVICE_PROCESS_REQUIREMENT)
                my_resource_requirements['RAM'] = eval(self.func_SERVICE_RAM_REQUIREMENT)
                my_resource_requirements['STORAGE'] = eval(self.func_SERVICE_STORAGE_REQUIREMENT)
                my_resource_requirements['PRIORITY'] = eval(self.func_SERVICE_PRIORITY)
                self.servicesResources[j] = my_resource_requirements

            self.appsResources.append(self.servicesResources)

            topologicorder_ = list(nx.topological_sort(APP))
            source = topologicorder_[0]

            self.appsSourceService.append(source)

            self.appsDeadlines[i] = self.myDeadlines[i]

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
                    myEdge['instructions'] = eval(self.func_SERVICEINSTR)
                    totalMIPS = totalMIPS + myEdge['instructions']
                    myEdge['bytes'] = eval(self.func_SERVICEMESSAGESIZE)
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
                myEdge['instructions'] = eval(self.func_SERVICEINSTR)
                totalMIPS = totalMIPS + myEdge['instructions']
                myEdge['bytes'] = eval(self.func_SERVICEMESSAGESIZE)
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
            probOfRequested = eval(self.func_REQUESTPROB)
            # probOfRequested = -1
            atLeastOneAllocated = False
            for j in self.gatewaysDevices:
                rand = random.random()
                if rand < probOfRequested:
                    myOneUser = {}
                    myOneUser['app'] = str(i)
                    myOneUser['message'] = "M.USER.APP." + str(i)
                    myOneUser['id_resource'] = j
                    myOneUser['lambda'] = eval(self.func_USERREQRAT)
                    userRequestList.add(j)
                    self.myUsers.append(myOneUser)
                    atLeastOneAllocated = True
            if not atLeastOneAllocated:
                j = random.randint(0, len(self.gatewaysDevices) - 1)
                myOneUser = {}
                myOneUser['app'] = str(i)
                myOneUser['message'] = "M.USER.APP." + str(i)
                myOneUser['id_resource'] = list(self.gatewaysDevices)[j]  # Random GW to host the request
                myOneUser['lambda'] = eval(self.func_USERREQRAT)
                userRequestList.add(list(self.gatewaysDevices)[j])
                self.myUsers.append(myOneUser)
            self.appsRequests.append(userRequestList)

        userJson['sources'] = self.myUsers

        userFile = open(self.resultFolder + "/usersDefinition.json", "w")
        userFile.write(json.dumps(userJson))
        userFile.close()

    def firstPlacement(self):
        servicesInFog = 0
        servicesInCloud = 0
        allAlloc = {}
        myAllocationList = list()
        # random.seed(datetime.now())
        initial_nodeResources = sorted(self.nodeResources.items(), key=operator.itemgetter(0))
        aux = sorted(self.nodeResources.items(), key=operator.itemgetter(1))
        # aux = sorted(self.nodeResources.items(), key=operator.itemgetter(0))
        sorted_nodeResources = [list(sub_list) for sub_list in aux]

        for app_num, app in zip(range(0, len(self.appsRequests)), self.appsRequests):
            for instance in range(0, len(self.appsRequests[app_num])):
                for module in list(self.apps[app_num].nodes):
                    flag = True
                    iterations = 0
                    while flag and iterations < (len(sorted_nodeResources) - 1):
                        # Chosing the node with less resources to host the service
                        index = iterations
                        iterations += 1
                        # Checking if the node has resource to host the service
                        res_required = self.servicesResources[module]
                        if res_required <= sorted_nodeResources[index][1]:
                            remaining_resources = sorted_nodeResources[index][1] - res_required
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
                    if iterations == (len(sorted_nodeResources) - 1):
                        print("After %i iterations it was not possible to place the module %i using the FirstPlacement" \
                              % iterations, module)
                        exit()
        allAlloc['initialAllocation'] = myAllocationList

        allocationFile = open(self.resultFolder + "/allocDefinitionFirst.json", "w")
        allocationFile.write(json.dumps(allAlloc))
        allocationFile.close()

        # Keeping nodes' resources
        final_nodeResources = sorted(self.nodeResources.items(), key=operator.itemgetter(0))
        # if os.stat('C:\\Users\\David Perez Abreu\\Sources\\Fog\\YAFS_Master\\src\\examples\\PopularityPlacement\\conf\\node_resources.csv').st_size == 0:
        #     # The file in empty
        #     ids = ['node_id']
        #     values = ['ini_resources']
        #     token = self.scenario + '_first'
        #     fvalues = [token]
        #     for ftuple in initial_nodeResources:
        #         ids.append(ftuple[0])
        #         values.append(ftuple[1])
        #     for stuple in final_nodeResources:
        #         fvalues.append(stuple[1])
        #     file = open('C:\\Users\\David Perez Abreu\\Sources\\Fog\\YAFS_Master\\src\\examples\\PopularityPlacement\\conf\\node_resources.csv', 'a+')
        #     file.write(",".join(str(item) for item in ids))
        #     file.write("\n")
        #     file.write(",".join(str(item) for item in values))
        #     file.write("\n")
        #     file.write(",".join(str(item) for item in fvalues))
        #     file.write("\n")
        #     file.close()
        # else:
        #     token = self.scenario + '_first'
        #     fvalues = [token]
        #     for stuple in final_nodeResources:
        #         fvalues.append(stuple[1])
        #     file = open(
        #         'C:\\Users\\David Perez Abreu\\Sources\\Fog\\YAFS_Master\\src\\examples\\PopularityPlacement\\conf\\node_resources.csv', 'a+')
        #     file.write(",".join(str(item) for item in fvalues))
        #     file.write("\n")
        #     file.close()

        print("First initial allocation performed!")


sg = ExperimentSetup(config=None)
sg.networkGeneration()
sg.appGeneration()
sg.userGeneration()
# sg.firstPlacement()
