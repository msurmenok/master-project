import random

from yafs.population import Population
from yafs.distribution import exponential_distribution, deterministic_distribution, uniformDistribution
from datetime import datetime



class JSONPopulation(Population):

    def __init__(self, json, iteration, **kwargs):
        super(JSONPopulation, self).__init__(**kwargs)
        self.data = json
        self.it = iteration

    def initial_allocation(self, sim, app_name):
        for item in self.data["sources"]:
            if item["app"] == app_name:
                app_name = item["app"]
                idtopo = item["id_resource"]
                lambd = item["lambda"]
                app = sim.apps[app_name]
                msg = app.get_message(item["message"])
                random.seed(datetime.now())

                dDistribution = exponential_distribution(name="Exp", lambd=random.randint(200, 1000), seed=self.it)
                # dDistribution = deterministic_distribution(time=200,name='DetD')  # experiment

                idsrc = sim.deploy_source(app_name, id_node=idtopo, msg=msg, distribution=dDistribution)
                random.randint(200, 1000)