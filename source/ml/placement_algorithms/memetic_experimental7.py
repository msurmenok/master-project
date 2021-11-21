import numpy as np

"""
Memetic algorithm below is based on C code from https://github.com/flopezpires/iMaVMP
"""


def initialize(num_creatures, num_services, num_hosts):
    # For each service randomly generate index of host where this service should be placed
    # population = np.zeros((num_creatures, num_services))
    # for i in range(0, num_creatures):
    #     for j in range(0, num_services):
    #         random_host_index = np.random.randint(low=0, high=num_hosts)
    #         population[i][j] = random_host_index

    population = np.random.randint(size=(num_creatures, num_services), low=0, high=num_hosts).astype(np.float64)
    return population


def check_feasibility_and_repair(services, individual, individual_hosts, MAX_PRIORITY):
    # change method signature, for each individual there will be a separate array of hosts,
    # add the third argument - other hosts that are not active rn but can be used
    # initialize to current hosts utilization, refresh for every creature
    hosts_utilization = np.array(individual_hosts, copy=True)
    hosts_utilization = np.delete(hosts_utilization, (3, 4, 5, 6), 1)
    num_services = individual.shape[0]
    feasible = True

    for i in range(num_services):
        # subtract the service resource requirements from current hosts utilization
        # if it become negative, meaning host overloading -> solution not feasible
        current_host_index = int(individual[i])
        if hosts_utilization[current_host_index][0] - services[i][0] < 0 or \
                hosts_utilization[current_host_index][1] - services[i][1] < 0 or \
                hosts_utilization[current_host_index][2] - services[i][2] < 0:
            feasible = False
            # if current service cannot be placed on this machine, set this service host index to Nan
            individual[i] = None
        else:
            hosts_utilization[current_host_index] = hosts_utilization[current_host_index] - services[i][:3]
    # if not feasible send the individual to repair
    if not feasible:
        return repair(services, individual, hosts_utilization, MAX_PRIORITY)
    else:
        return individual, np.copy(hosts_utilization)


def repair(services, individual, hosts_utilization, MAX_PRIORITY):
    """repair any violation of our constraints
    (e.g., services to be hosted on a device with resource consumption
    greater than available capacity of the volunteer"""
    """
    iterate over services placement
    if nan, try to place it into a random host
    if cannot be placed into a random host twice, check if service has highest priority
    if not, just remove (?) service
    else try to place service to another available host
    """
    num_services = len(services)
    for i in range(num_services):
        # try to move service to an available host
        if np.isnan(individual[i]):
            fixed = False  # flag to indicate if the service was sucessfully placed when iterating over all hosts
            # try to place the service on one of the available hosts
            # start with the random host
            candidate = np.random.randint(low=0, high=len(hosts_utilization))
            for j in range(len(hosts_utilization)):
                candidate_index = (candidate + j) % len(hosts_utilization)  # make sure that host index is in range
                # check if candidate host has enough resources
                if fixed:
                    break
                if hosts_utilization[candidate_index][0] - services[i][0] > 0 and \
                        hosts_utilization[candidate_index][1] - services[i][1] > 0 and \
                        hosts_utilization[candidate_index][2] - services[i][2] > 0:
                    # if enough resources, assign service to this host and switch fixed flag to True
                    # and update host utilization
                    individual[i] = candidate_index
                    fixed = True
                    hosts_utilization[candidate_index][0] = hosts_utilization[candidate_index][0] - services[i][0]
                    hosts_utilization[candidate_index][1] = hosts_utilization[candidate_index][1] - services[i][1]
                    hosts_utilization[candidate_index][2] = hosts_utilization[candidate_index][2] - services[i][2]
                # else continue iterating to search for suitable host
            # out of for loop, didn't find a suitable host among running volunteers, keep it Nan meaninng that it should go to cloud
    # return repaired individual
    return (individual, np.copy(hosts_utilization))


def fitness(individual, services, hosts, user_to_host_distance, distance_to_cloud, max_priority, min_deadline,
            max_deadline):
    priority_index = 3  # the last index (3) in services describes its priority
    time_index = 3  # index 3 in host describes how long it is available
    deadline_diff = 1
    if max_deadline - min_deadline > 0:
        deadline_diff = max_deadline - min_deadline

    # MAX, F1 - Number of Pushed Services Maximization
    f1 = 0
    for i in range(len(individual)):
        if not np.isnan(individual[i]):
            # if the service host index is not Nan, then this service was placed sucessfully
            f1 += 1

    # MAX, F2 - QoS Maximization: maximum number of services with high priorities are pushed
    # F2 = sum of all servies i (Cpi * Pi * Ri),
    # Cpi - constant that prioritize services with high P i over others with low value
    # c_pi = 10  # constant set to 10 in paper description
    c_pi = 1
    f2 = 0

    for i in range(len(individual)):
        p_i = 1 if (services[i][priority_index] == max_priority) else 0
        if not np.isnan(individual[i]):
            # if the service host index is not Nan, then this service was placed sucessfully
            f2 += c_pi * p_i

    # MAX, F3 - Survivability Factor Maximization
    # Maximizing the time a device is available to host particular services
    # sum of all hosts time availability that hosts at least one service
    active_hosts_indexes = np.unique(individual)
    active_hosts_indexes = [x for x in active_hosts_indexes if not np.isnan(x)]
    f3 = 0
    # for host_index in active_hosts_indexes:
    #     f3 += hosts[int(host_index)][time_index]

    for i in range(len(individual)):
        if not np.isnan(individual[i]):
            f3 += hosts[int(individual[i])][time_index]

    # MIN, F4
    # Host Distance Minimization
    f4 = 0

    for i in range(len(individual)):
        if not np.isnan(individual[i]):
            f4 += user_to_host_distance[i][int(individual[i])]
        else:
            f4 += distance_to_cloud

    # MIN, F5
    # Active Hosts Minimization
    f5 = len(active_hosts_indexes)

    # service: [cpu, ram, storage, priority, x, y, deadline]
    # host: [cpu, ram, storage, time_availability, x, y, ipt]

    # MAX, F6
    # Prioritize placement of services with smallest deadline on fog
    # normalize deadlines: value - min_deadline / (max_deadline - min_deadline)
    # inverse it, 1 - result from previous step
    f6 = 0
    deadline_index = 6
    for i in range(len(individual)):
        if not np.isnan(individual[i]):
            normalized_deadline = (services[i][deadline_index] - min_deadline) / deadline_diff
            value = (1 - normalized_deadline)
            f6 += value

    # MAX, F7
    # Prioritize hosts with fastest processors, higher IPT
    f7 = 0
    ipt_index = 6
    for i in range(len(individual)):
        if not np.isnan(individual[i]):
            f7 += hosts[int(individual[i])][ipt_index]

    objectives = np.array([f1, f2, f3, f4, f5, f6, f7])
    return objectives


# local search
def local_search(population, utilization, hosts, services, number_of_individuals, h_size, s_size,
                 service_to_closest_host):
    iterator_individual = 0
    physical_position = 0
    physical_position2 = 0
    iterator_position = 0
    iterator_virtual = 0
    host_id = 0
    iterator_physical = 0
    option = 0

    option_to_execute = 0
    val_rand = np.random.uniform(low=0.0, high=1.0)

    if val_rand > 0 and val_rand <= 0.5:
        option_to_execute = 0
    else:
        option_to_execute = 3

    while option < 2:
        if val_rand > 0 and val_rand <= 0.5:
            option_to_execute += 1
            option += 1
        elif val_rand > 0.5 and val_rand < 1.0:
            option_to_execute -= 1
            option += 1

        if option_to_execute == 1:
            for iterator_individual in range(number_of_individuals):
                for iterator_virtual in range(s_size):
                    physical_position = population[iterator_individual][iterator_virtual]
                    if not np.isnan(physical_position):
                        physical_position = int(physical_position)
                        for host_id in range(s_size):
                            physical_position2 = population[iterator_individual][host_id]
                            if not np.isnan(physical_position2):
                                physical_position2 = int(physical_position2)
                                if physical_position != physical_position2:
                                    if utilization[iterator_individual][physical_position][0] - \
                                            services[host_id][0] >= 0 \
                                            and utilization[iterator_individual][physical_position][1] - \
                                            services[host_id][1] >= 0 \
                                            and utilization[iterator_individual][physical_position][2] - \
                                            services[host_id][2] >= 0:
                                        utilization[iterator_individual][physical_position2][0] += \
                                            services[host_id][0]
                                        utilization[iterator_individual][physical_position2][1] += \
                                            services[host_id][1]
                                        utilization[iterator_individual][physical_position2][2] += \
                                            services[host_id][2]

                                        utilization[iterator_individual][physical_position][0] -= \
                                            services[host_id][0]
                                        utilization[iterator_individual][physical_position][1] -= \
                                            services[host_id][1]
                                        utilization[iterator_individual][physical_position][2] -= \
                                            services[host_id][2]
                                        population[iterator_individual][host_id] = \
                                            population[iterator_individual][iterator_virtual]

        if option_to_execute == 2:
            for iterator_individual in range(number_of_individuals):
                for iterator_virtual in range(s_size):
                    physical_position = population[iterator_individual][iterator_virtual]
                    if np.isnan(physical_position):
                        for host_id in range(len(hosts)):
                            physical_position2 = service_to_closest_host[iterator_virtual][host_id]
                            physical_position2 = int(physical_position2)
                            if utilization[iterator_individual][physical_position2][0] - \
                                    services[iterator_virtual][0] >= 0 \
                                    and utilization[iterator_individual][physical_position2][1] - \
                                    services[iterator_virtual][1] >= 0 \
                                    and utilization[iterator_individual][physical_position2][2] - \
                                    services[iterator_virtual][2] >= 0:
                                utilization[iterator_individual][physical_position2][0] -= \
                                    services[iterator_virtual][0]
                                utilization[iterator_individual][physical_position2][1] -= \
                                    services[iterator_virtual][1]
                                utilization[iterator_individual][physical_position2][2] -= \
                                    services[iterator_virtual][2]
                                population[iterator_individual][iterator_virtual] = population[iterator_individual][
                                    host_id]
                                break
    return population, utilization


def non_dominated_sorting(solutions, number_of_individuals):
    #     iterator_solution = 0
    #     iterator_comparison = 0
    actual_pareto_front = 1
    solutions_allocated = 0
    pareto_fronts = np.zeros((number_of_individuals))
    # auxiliar integers
    dominance = 0
    dont_add = 0
    allocated_solutions = 0
    # while all the solutions have been evaluated
    while allocated_solutions < number_of_individuals:
        # iterate on solutions
        for iterator_solution in range(number_of_individuals):
            # flag for a solution to be added
            dont_add = 0
            if pareto_fronts[iterator_solution] == 0:
                for iterator_comparision in range(number_of_individuals):
                    # if the solution is not itself, it is not been evaluated or is in the actual Pareto front
                    if iterator_solution != iterator_comparision and pareto_fronts[iterator_comparision] == 0 or \
                            pareto_fronts[iterator_comparision] == actual_pareto_front:
                        # verificate the dominance between both
                        dominance = is_dominated(solutions, iterator_solution, iterator_comparision)
                        # is dominated by a solution that is in the Pareto front, so this solution is not added
                        if dominance:
                            dont_add = 1
                            break
                # if the solution is not dominated by any other, let's add it to the actual Pareto front
                if dont_add == 0:
                    pareto_fronts[iterator_solution] = actual_pareto_front
                    allocated_solutions += 1
        actual_pareto_front += 1
    return pareto_fronts


def is_dominated(solution, a, b):
    # if b dominates a
    # otherwise return 0 meaning that a either dominates b or have the same dominance
    # 0, 1, 2 - maximization functions, 3 and 4 - minimization functions
    # if solution[b][0] >= solution[a][0] and solution[b][1] >= solution[a][1] and solution[b][2] >= solution[a][2] and \
    #         solution[b][3] <= solution[a][3] and solution[b][4] < solution[a][4] or \
    #         solution[b][0] >= solution[a][0] and solution[b][1] >= solution[a][1] and solution[b][2] >= solution[a][
    #     2] and solution[b][3] < solution[a][3] and solution[b][4] <= solution[a][4] or \
    #         solution[b][0] >= solution[a][0] and solution[b][1] >= solution[a][1] and solution[b][2] > solution[a][
    #     2] and solution[b][3] <= solution[a][3] and solution[b][4] <= solution[a][4] or \
    #         solution[b][0] >= solution[a][0] and solution[b][1] > solution[a][1] and solution[b][2] >= solution[a][
    #     2] and solution[b][3] <= solution[a][3] and solution[b][4] <= solution[a][4] or \
    #         solution[b][0] > solution[a][0] and solution[b][1] >= solution[a][1] and solution[b][2] >= solution[a][
    #     2] and solution[b][3] <= solution[a][3] and solution[b][4] <= solution[a][4]:
    #     return -1
    # return 0
    cond1 = solution[b][0] >= solution[a][0] and solution[b][1] >= solution[a][1] and solution[b][2] >= solution[a][
        2] and \
            solution[b][3] <= solution[a][3] and solution[b][4] <= solution[a][4] and solution[b][5] >= solution[a][
                5] and solution[b][6] >= solution[a][6]
    cond2 = solution[b][0] == solution[a][0] and solution[b][1] == solution[a][1] and solution[b][2] == solution[a][
        2] and solution[b][3] == solution[a][3] and solution[b][4] == solution[a][4] and solution[b][5] == solution[a][
                5] and solution[b][6] == solution[a][6]
    return cond1 and (not cond2)


class Pareto_element:
    def __init__(self, solution, cost):
        self.solution = solution
        self.cost = cost

    def __str__(self):
        return str(self.solution)

    def __repr__(self):
        return str(self.solution)

    def __eq__(self, other):
        return np.array_equal(self.solution, other.solution)


def pareto_insert(pareto_head, individual, objectives_functions):
    pareto_element = Pareto_element(solution=individual, cost=objectives_functions)
    if pareto_element not in pareto_head:
        pareto_head.append(pareto_element)
    return pareto_head


def get_min_cost(pareto_head):
    pass


def get_max_cost(pareto_head):
    pass


def selection(fronts, number_of_individuals, percent):
    possible_parent = 0
    # generate randomically a parent candidate
    actual_parent = np.random.randint(low=0, high=number_of_individuals)
    # iterate on positions of an individual and select the parents for the crossover
    for i in range(int(number_of_individuals * percent)):
        posible_parent = np.random.randint(low=0, high=number_of_individuals)
        if (fronts[actual_parent] > fronts[posible_parent]):
            actual_parent = posible_parent
    return actual_parent


def crossover(population, position_parent1, position_parent2, num_services):
    crossover_point = int(num_services / 2)
    aux = population[position_parent1][crossover_point:]
    population[position_parent1][crossover_point:] = population[position_parent2][crossover_point:]
    population[position_parent2][crossover_point:] = aux
    return population


def mutation(population, services, num_individuals, num_hosts, num_services):
    for i in range(num_individuals):
        for j in range(num_services):
            probability = np.random.uniform(low=0.0, high=1.0)
            # if the probablidad is less than 1/v_size, performs the mutation
            if probability < 1 / num_services:
                new_position = np.random.randint(low=0, high=num_hosts)
                population[i][j] = new_position
    return population


def load_utilization(population, hosts, services, number_of_individuals, h_size, s_size):
    # utilization is the remained resources on a host after placing service on it
    utilization = np.zeros((number_of_individuals, h_size, 3))
    requirements = np.zeros((h_size, 3))

    for i in range(number_of_individuals):
        # requirements in Processor, Memory and Storage. Initialized to 0
        requirements = np.zeros((h_size, 3))
        for j in range(s_size):
            # if the virtual machine has a placement assigned
            if not np.isnan(population[i][j]):
                requirements[int(population[i][j])][0] += services[j][0]
                requirements[int(population[i][j])][1] += services[j][1]
                requirements[int(population[i][j])][2] += services[j][2]
        utilization[i] = hosts[:, :3] - requirements
        # for j in range(h_size):
        #     utilization[i][j][0] = hosts[j][0] - requirements[j][0]
        #     utilization[i][j][1] = hosts[j][1] - requirements[j][1]
        #     utilization[i][j][2] = hosts[j][2] - requirements[j][2]
    return utilization


# population_evolution: update the pareto front in the population
def population_evolution(P, Q, objectives_functions_P, objectives_functions_Q, fronts_P, number_of_individuals,
                         num_services):
    # copy the P individual and objective function
    # copy the Q individual and objective function
    PQ = np.concatenate((P, Q), axis=0)
    objectives_functions_PQ = np.concatenate((objectives_functions_P, objectives_functions_Q), axis=0)

    # calculate fitness
    fronts_PQ = non_dominated_sorting(objectives_functions_PQ, number_of_individuals * 2)

    iterator_P = 0
    actual_pareto = 0
    while iterator_P < number_of_individuals:
        actual_pareto += 1
        for iterator in range(number_of_individuals * 2):
            if fronts_PQ[iterator] == actual_pareto and iterator_P < number_of_individuals:
                if objectives_functions_PQ[iterator][0] != 0 or objectives_functions_PQ[iterator][1] != 0 or \
                        objectives_functions_PQ[iterator][2] != 0 or objectives_functions_PQ[iterator][2] != 0 or \
                        objectives_functions_PQ[iterator][3] != 0 or objectives_functions_PQ[iterator][4] != 0 or \
                        objectives_functions_PQ[iterator][5] != 0 or objectives_functions_PQ[iterator][6] != 0:
                    objectives_functions_P[iterator_P][0] = objectives_functions_PQ[iterator][0]
                    objectives_functions_P[iterator_P][1] = objectives_functions_PQ[iterator][1]
                    objectives_functions_P[iterator_P][2] = objectives_functions_PQ[iterator][2]
                    objectives_functions_P[iterator_P][3] = objectives_functions_PQ[iterator][3]
                    objectives_functions_P[iterator_P][4] = objectives_functions_PQ[iterator][4]
                    objectives_functions_P[iterator_P][5] = objectives_functions_PQ[iterator][5]
                    objectives_functions_P[iterator_P][6] = objectives_functions_PQ[iterator][6]
                    fronts_P[iterator_P] = fronts_PQ[iterator]
                    P[iterator_P] = PQ[iterator]
                    iterator_P += 1
    return P


# based on memetic experimental 2
def memetic_experimental7(num_creatures, NUM_GENERATIONS, services, hosts, MAX_PRIORITY, distance_to_cloud):
    num_services = len(services)
    num_hosts = len(hosts)
    num_objective_functions = 7
    SELECTION_PERCENT = 0.5

    # calculate distance between each user (who requested service) and host using x, y coordinates
    # row (outer index) = service id, column (inner index) = host id
    user_to_host_distance = np.zeros((num_services, num_hosts))

    for i in range(num_services):
        user_coordinates = np.array((services[i][4], services[i][5]))  # 4 is index for x, 5 is index for y
        for j in range(num_hosts):
            host_coordinates = np.array((hosts[j][4], hosts[j][5]))
            distance = np.linalg.norm(user_coordinates - host_coordinates)
            user_to_host_distance[i][j] = distance

    # create hashmap, service id to sorted array of hosts by distance
    # service_id: [host_id1, host_id2, host_id3]
    service_to_closest_host = dict()
    for i in range(len(user_to_host_distance)):
        service_to_closest_host[i] = np.argsort(user_to_host_distance[i])

    P = initialize(num_creatures, num_services, num_hosts)

    # repair
    repaired_population = np.zeros((num_creatures, num_services))
    hosts_utilization_for_each_creature = np.zeros(
        (num_creatures, num_hosts, 3))  # 3 because we have 3 kinds of resources
    for i in range(num_creatures):
        result = check_feasibility_and_repair(services, P[i], hosts, MAX_PRIORITY)
        repaired_population[i] = result[0]
        hosts_utilization_for_each_creature[i] = result[1]

    # apply local search to solutions
    P, hosts_utilization_for_each_creature = local_search(repaired_population, hosts_utilization_for_each_creature,
                                                          hosts, services, num_creatures, num_hosts, num_services,
                                                          service_to_closest_host)

    # parameters to normalize fitness function
    priority_index = 3  # the last index (3) in services describes its priority
    deadline_index = 6  # index 3 in host describes how long it is available
    num_important_services = services[(services[:, priority_index] == MAX_PRIORITY)]
    num_important_services, _ = num_important_services.shape
    services_max = (services.max(axis=0))
    services_min = (services.min(axis=0))
    deadline_max = services_max[deadline_index]
    deadline_min = services_min[deadline_index]

    # calculate the cost of each objective function for each solution
    objectives_functions_P = np.zeros((num_creatures, num_objective_functions))
    for i in range(num_creatures):
        fitness_score = fitness(P[i], services, hosts, user_to_host_distance, distance_to_cloud, MAX_PRIORITY,
                                deadline_min, deadline_max)
        objectives_functions_P[i] = fitness_score

    # calculate the non-dominated fronts
    fronts_P = non_dominated_sorting(objectives_functions_P, num_creatures)

    # Update set of nondominated solutions
    pareto_head = list()
    for i in range(num_creatures):
        if fronts_P[i] == 1:
            pareto_head = pareto_insert(pareto_head, P[i], objectives_functions_P[i])

    # identificators for the crossover parents
    father = []
    mother = []
    Q = initialize(num_creatures, num_services, num_hosts)
    utilization_Q = load_utilization(Q, hosts, services, num_creatures, num_hosts, num_services);
    objectives_functions_Q = np.zeros((num_creatures, num_objective_functions))
    fronts_Q = []

    generation = 0
    while (generation < NUM_GENERATIONS):
        generation += 1

        Q = initialize(num_creatures, num_services, num_hosts)
        father = selection(fronts_P, num_creatures, SELECTION_PERCENT)
        mother = selection(fronts_P, num_creatures, SELECTION_PERCENT)

        # crossover and mutation of solutions
        Q = crossover(Q, father, mother, num_services)
        Q = mutation(Q, services, num_creatures, num_hosts, num_services)
        # load the utilization of physical machines and network links of all individuals/solutions
        utilization_Q = load_utilization(Q, hosts, services, num_creatures, num_hosts, num_services);

        # repair
        for i in range(num_creatures):
            result = check_feasibility_and_repair(services, Q[i], hosts, MAX_PRIORITY)
            Q[i] = result[0]
            utilization_Q[i] = result[1]

        Q, utilization_Q = local_search(Q, utilization_Q, hosts, services, num_creatures, num_hosts, num_services,
                                        service_to_closest_host)

        # calculate the cost of each objective function for each solution
        for i in range(num_creatures):
            fitness_score = fitness(Q[i], services, hosts, user_to_host_distance, distance_to_cloud, MAX_PRIORITY,
                                    deadline_min, deadline_max)
            objectives_functions_Q[i] = fitness_score
        # calculate the non-dominated fronts
        fronts_Q = non_dominated_sorting(objectives_functions_Q, num_creatures)
        # Update set of nondominated solutions Pc from Qt
        for i in range(num_creatures):
            if fronts_Q[i] == 1:
                pareto_head = pareto_insert(pareto_head, Q[i], objectives_functions_Q[i])
        #     Pt = fitness selection from Pt ∪ Qt
        P = population_evolution(P, Q, objectives_functions_P, objectives_functions_Q, fronts_P, num_creatures,
                                 num_services)
    # print("final P", P)
    # print("final utilization", load_utilization(P, hosts, services, num_creatures, num_hosts, num_services))

    # report the best
    # cost_solution = report_best_population(P, distance_to_cloud, hosts, num_creatures, num_objective_functions,
    #                                        services, user_to_host_distance, MAX_PRIORITY)
    return report_best_population(pareto_head, hosts, services, num_hosts, num_services)


def report_best_population(pareto_head, hosts, services, h_size, s_size):
    pareto_size = len(pareto_head)

    best_P = list()
    objective_functions_best_P = list()

    for pareto_element in pareto_head:
        objective_function = pareto_element.cost
        solution = pareto_element.solution
        objective_functions_best_P.append(objective_function)
        best_P.append(solution)

    objective_functions_best_P = np.stack(objective_functions_best_P)
    objectives_max = objective_functions_best_P.max(axis=0)
    objectives_min = objective_functions_best_P.min(axis=0)
    objectives_diff = objectives_max - objectives_min
    w1 = 1 / 7
    w2 = 1 / 7
    w3 = 1 / 7
    w4 = 1 / 7
    w5 = 1 / 7
    w6 = 1 / 7
    w7 = 1 / 7

    best_P = np.stack(best_P)

    fronts_best_P = non_dominated_sorting(objective_functions_best_P, pareto_size)
    utilization_best_P = load_utilization(best_P, hosts, services, pareto_size, h_size, s_size)

    solution_objf = list()

    # all solutions
    for i in range(pareto_size):
        solution_objf.append((best_P[i], objective_functions_best_P[i]))

    # find the best solution
    cost_solution_objf = list()
    for i in range(pareto_size):
        if (fronts_best_P[i] == 1):
            solution = best_P[i]
            obj_f = objective_functions_best_P[i]

            obj_f_normalized = np.zeros(len(obj_f))
            for k in range(len(obj_f_normalized)):
                if objectives_diff[k] != 0:
                    obj_f_normalized[k] = (obj_f[k] - objectives_min[k]) / objectives_diff[k]
                else:
                    obj_f_normalized[k] = (obj_f[k] - objectives_min[k])

            cost = w1 * obj_f_normalized[0] + w2 * obj_f_normalized[1] + w3 * obj_f_normalized[2] - w4 * \
                   obj_f_normalized[3] - w5 * obj_f_normalized[4] + w6 * obj_f_normalized[5] + w7 * obj_f_normalized[6]
            cost_solution_objf.append((solution, cost, obj_f))
    best_solution = sorted(cost_solution_objf, key=lambda x: x[1], reverse=True)
    best_solution = best_solution[0]
    result = (best_solution[0], best_solution[2])

    return result, solution_objf


def test_memetic():
    # test memetic
    # print("Test memetic")
    # initialize services
    # service | CPU | MEM | DISK | PRIORITY | deadline
    services = np.array([[0.12, 0.2, 0.2, 0, 10, 200, 5000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 5000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 2000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 23000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 15000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 5000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 5000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 1500],
                         [0.18, 0.20, 0.32, 0, 200, 50, 4000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 4000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 5000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 5000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 15000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 15000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 15000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 15000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 5000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 8000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 8000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 9000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 9000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 15000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 2000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 5000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 15000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 4000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 3000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 5000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 5000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 10000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 10000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 10000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 10000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 10000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 10000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 10000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 10000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 10000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 15000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 15000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 15000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 15000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 15000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 15000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 15000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 15000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 15000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 15000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 18000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 18000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 18000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 18000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 18000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 18000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 18000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 18000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 18000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 18000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 18000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 18000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 18000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 18000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 18000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 18000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 18000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 18000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 18000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 18000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 18000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 18000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 18000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 18000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 18000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 18000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 18000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 18000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 18000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 18000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 20000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 20000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 20000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 20000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 20000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 20000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 20000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 20000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 20000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 20000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 20000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 20000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 20000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 20000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 20000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 20000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 20000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 20000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 20000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 20000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 10000],
                         [0.24, 0.23, 0.1, 1, 300, 400, 10000],
                         [0.18, 0.20, 0.32, 0, 200, 50, 10000],
                         [0.12, 0.2, 0.2, 0, 10, 200, 10000],
                         ])

    # print(services)

    # initialize hosts
    # host           | CPU | MEM | DISK | TIME | DISTANCE | IPT
    hosts = np.array([[0.5, 0.24, 0.4, 2000, 500, 500, 500],
                      [0.25, 0.4, 0.4, 500, 50, 50, 500],
                      [0.25, 0.26, 0.2, 300, 40, 50, 500],
                      [0.8, 0.24, 0.38, 1000, 25, 900, 500],
                      [0.2, 0.76, 0.62, 70, 50, 400, 500],
                      [0.6, 0.6, 0.5, 60, 1000, 30, 500],
                      [1, 1, 1, 65, 20, 800, 700],
                      [0.5, 0.24, 0.4, 2000, 500, 500, 700],
                      [0.25, 0.4, 0.4, 500, 50, 50, 700],
                      [0.25, 0.26, 0.2, 300, 40, 50, 700],
                      [0.8, 0.24, 0.38, 1000, 25, 900, 700],
                      [0.2, 0.76, 0.62, 70, 50, 400, 700],
                      [0.6, 0.6, 0.5, 60, 1000, 30, 700],
                      [1, 1, 1, 65, 20, 800, 700],
                      [0.5, 0.24, 0.4, 2000, 500, 500, 700],
                      [0.25, 0.4, 0.4, 500, 50, 50, 700],
                      [0.25, 0.26, 0.2, 300, 40, 50, 700],
                      [0.8, 0.24, 0.38, 1000, 25, 900, 700],
                      [0.2, 0.76, 0.62, 70, 50, 400, 700],
                      [0.6, 0.6, 0.5, 60, 1000, 30, 700],
                      [1, 1, 1, 65, 20, 800, 700],
                      [0.5, 0.24, 0.4, 2000, 500, 500, 700],
                      [0.25, 0.4, 0.4, 500, 50, 50, 700],
                      [0.25, 0.26, 0.2, 300, 415, 50, 700],
                      [0.8, 0.24, 0.38, 1000, 25, 900, 700],
                      [0.2, 0.76, 0.62, 70, 50, 400, 700],
                      [0.6, 0.6, 0.5, 60, 1000, 30, 700],
                      [1, 1, 1, 65, 20, 800, 800],
                      [0.5, 0.24, 0.4, 2000, 500, 500, 800],
                      [0.25, 0.4, 0.4, 500, 50, 50, 800],
                      [0.25, 0.26, 0.2, 300, 40, 50, 800],
                      [0.8, 0.24, 0.38, 1000, 25, 900, 800],
                      [0.2, 0.76, 0.62, 70, 50, 400, 800],
                      [0.6, 0.6, 0.5, 60, 1000, 30, 800],
                      [1, 1, 1, 65, 20, 800, 800],
                      [0.6, 0.6, 0.5, 60, 1000, 30, 800],
                      [1, 1, 1, 65, 20, 800, 800],
                      [0.5, 0.24, 0.4, 2000, 500, 500, 800],
                      [0.25, 0.4, 0.4, 500, 50, 50, 800],
                      [0.25, 0.26, 0.2, 300, 40, 50, 800],
                      [0.8, 0.24, 0.38, 1000, 25, 900, 800],
                      [0.2, 0.76, 0.62, 70, 50, 400, 800],
                      [0.6, 0.6, 0.5, 60, 1000, 30, 800],
                      [1, 1, 1, 65, 20, 800, 850],
                      [0.5, 0.24, 0.4, 2000, 500, 500, 850],
                      [0.25, 0.4, 0.4, 500, 50, 50, 850],
                      [0.25, 0.26, 0.2, 300, 40, 50, 900],
                      [0.8, 0.24, 0.38, 1000, 25, 900, 950],
                      [0.2, 0.76, 0.62, 70, 50, 400, 1000],
                      ])

    # introduce some hosts that are not currently running. For now just a copy of available hosts

    # all together
    # constants here
    NUM_GENERATIONS = 100
    num_creatures = 20
    # num_services = 3
    # num_hosts = 5  # number of hosts may be different if we add extra during repair
    MAX_PRIORITY = 1  # max priority can be 0 or 1
    best_placement, placement = memetic_experimental7(num_creatures, NUM_GENERATIONS, services, hosts, MAX_PRIORITY, 18200)
    print("Best placement: ", best_placement)


# test_memetic()


def doprofiling():
    import cProfile, pstats
    import time

    profiler = cProfile.Profile()

    s_time = time.time()
    profiler.enable()
    test_memetic()
    profiler.disable()

    total_time = time.time() - s_time

    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('profiler_without_localsearch_cumtime')
    stats.print_stats()
    print('------ by total number')
    stats2 = pstats.Stats(profiler).sort_stats('tottime')
    stats2.dump_stats('profiler_without_localsearch_tottime')
    stats2.print_stats()

    print('==========================')
    print("total time = ", total_time)

# doprofiling()
