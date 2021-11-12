import json

def run_new_scenario(configuration,iteration):
    # take information about resources from
    data_folder = "data/data_" + configuration['scenario'] + '_' + str(iteration)


    # read netDefinition to get information about available resources
    netDefinitionFile = open(data_folder + '/netDefinition.json')
    netDefinition = json.load(netDefinitionFile)
    fogDevicesInfo = netDefinition['entity']
    # form netDefinition get all entities that are not gateways

    # read appDefinition to get information about applications
    appDefinitionFile = open(data_folder + '/appDefinition.json')
    appDefinition = json.load(appDefinitionFile)
    # retrieve deadline, module[0]['id or name'], module[0]['CPU'], RAM, STORAGE, PRIORITY

    # to retrieve source ids, and users location we do not use gateways in placement
    usersDefinitionFile = open(data_folder + '/usersDefinition.json')
    usersDefnition = json.load(usersDefinitionFile)

    print("done")


run_new_scenario({'scenario': 'small'}, 0)