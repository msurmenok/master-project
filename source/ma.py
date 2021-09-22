import numpy as np


def initialize(num_creatures, num_services, num_hosts):
    # For each service randomly generate index of host where this service should be placed
    population = np.zeros((num_creatures, num_services))
    for i in range(0, num_creatures):
        for j in range(0, num_services):
            random_host_index = np.random.randint(low=0, high=num_hosts)
            population[i][j] = random_host_index
    return population


def check_feasibility_and_repair(services, individual, individual_hosts, MAX_PRIORITY):
    # change method signature, for each individual there will be a separate array of hosts,
    # add the third argument - other hosts that are not active rn but can be used
    # initialize to current hosts utilization, refresh for every creature
    hosts_utilization = np.array(individual_hosts, copy=True)
    hosts_utilization = np.delete(hosts_utilization, (3, 4, 5), 1)
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
            #  print("not feasible", current_host_index)
            # if current service cannot be placed on this machine, set this service host index to Nan
            individual[i] = None
        else:
            hosts_utilization[current_host_index] = hosts_utilization[current_host_index] - services[i][:3]
    # if not feasible send the individual to repair
    if not feasible:
        #         print("Not feasible")
        #         print("if nan, service need to be reassigned: ", individual)
        #         print("FIXING IN PROGRESS:")
        return repair(services, individual, hosts_utilization, MAX_PRIORITY)
    else:
        #         print("Feasible!", individual)
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
            #             print("IS NAN ", i)
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


def fitness(individual, services, hosts):
    # MAX, F1 - Number of Pushed Services Maximization
    f1 = 0
    for i in range(len(individual)):
        if not np.isnan(individual[i]):
            # if the service host index is not Nan, then this service was placed sucessfully
            f1 += 1

    # MAX, F2 - QoS Maximization: maximum number of services with high priorities are pushed
    # F2 = sum of all servies i (Cpi * Pi * Ri),
    # Cpi - constant that prioritize services with high P i over others with low value
    c_pi = 10  # constant set to 10 in paper description
    f2 = 0
    priority_index = 3  # the last index (3) in services describes its priority
    for i in range(len(individual)):
        p_i = services[i][priority_index]
        #         print("priority P(i) = ", p_i)
        if not np.isnan(individual[i]):
            # if the service host index is not Nan, then this service was placed sucessfully
            f2 += c_pi * p_i

    # MAX, F3 - Survivability Factor Maximization
    # Maximizing the time a device is available to host particular services
    # sum of all hosts time availability that hosts at least one service
    time_index = 3  # index 3 in host describes how long it is available
    active_hosts_indexes = np.unique(individual)
    active_hosts_indexes = [x for x in active_hosts_indexes if not np.isnan(x) == True]
    f3 = 0
    for host_index in active_hosts_indexes:
        f3 += hosts[int(host_index)][time_index]

    # TODO: replace comparing distances (x,y) between user and host
    # MIN, F4
    # Host Distance Minimization
    location_index = 4  # index 4 in host describes the location of the host
    f4 = 0
    for host_index in active_hosts_indexes:
        f4 += hosts[int(host_index)][location_index]

    # MIN, F5
    # Active Hosts Minimization
    f5 = len(active_hosts_indexes)

    # weights, sum of all weights should be equal 1:
    w1 = w2 = w3 = w4 = w5 = 0.2
    # add all max objectives, subtract all min objectives
    objectives = np.array([(f1 * w1), (f2 * w2), (f3 * w3), (f4 * w4), (f5 * w5)])
    return objectives


# local search
def local_search(population, utilization, hosts, services, number_of_individuals, h_size, s_size):
    for i in range(number_of_individuals):
        random_number = np.random.uniform(low=0.0, high=1.0)
        if random_number > 0.5:
            population[i], utilization[i] = minimize_running_hosts(population[i], utilization[i], hosts, services,
                                                                   number_of_individuals, h_size, s_size)
            population[i], utilization[i] = maximize_running_services(population[i], utilization[i], hosts, services,
                                                                      number_of_individuals, h_size, s_size)
        else:
            population[i], utilization[i] = minimize_running_hosts(population[i], utilization[i], hosts, services,
                                                                   number_of_individuals, h_size, s_size)
            population[i], utilization[i] = maximize_running_services(population[i], utilization[i], hosts, services,
                                                                      number_of_individuals, h_size, s_size)
    return population, utilization


# helper function for local search
def minimize_running_hosts(creature, hosts_utilization_for_creature, hosts, services, number_of_individuals, h_size,
                           s_size):
    # iterate over services (placement) trying to move services to the same host
    for i in range(s_size):
        physical_position = creature[i]
        if not np.isnan(physical_position):
            physical_position = int(creature[i])
            # iterate over other hosts
            for j in range(s_size):
                physical_position_2 = creature[j]
                if not np.isnan(physical_position_2) and physical_position != physical_position_2:
                    # check if there's enough resources to migration the second service into
                    # the host where first service is placed
                    physical_position_2 = int(creature[j])
                    if hosts_utilization_for_creature[physical_position][0] - services[j][0] > 0 and \
                            hosts_utilization_for_creature[physical_position][1] - services[j][1] > 0 and \
                            hosts_utilization_for_creature[physical_position][2] - services[j][2] > 0:
                        # migrate service j
                        creature[j] = physical_position
                        # update utilization
                        hosts_utilization_for_creature[physical_position][0] = \
                            hosts_utilization_for_creature[physical_position][0] - services[j][0]
                        hosts_utilization_for_creature[physical_position][1] = \
                            hosts_utilization_for_creature[physical_position][1] - services[j][1]
                        hosts_utilization_for_creature[physical_position][2] = \
                            hosts_utilization_for_creature[physical_position][2] - services[j][2]

                        hosts_utilization_for_creature[physical_position_2][0] = \
                            hosts_utilization_for_creature[physical_position_2][0] + services[j][0]
                        hosts_utilization_for_creature[physical_position_2][1] = \
                            hosts_utilization_for_creature[physical_position_2][1] + services[j][1]
                        hosts_utilization_for_creature[physical_position_2][2] = \
                            hosts_utilization_for_creature[physical_position_2][2] + services[j][2]
    return creature, hosts_utilization_for_creature


# second helper function for local search
def maximize_running_services(creature, hosts_utilization_for_creature, hosts, services, number_of_individuals, h_size,
                              s_size):
    for i in range(s_size):
        physical_position = creature[i]
        if np.isnan(physical_position):
            # iterate over other hosts
            for j in range(s_size):
                physical_position_2 = creature[j]
                if not np.isnan(physical_position_2):
                    physical_position_2 = int(creature[j])
                    # If the use of the VM not exceeds the capacity of the physical machine performs the migration
                    if hosts_utilization_for_creature[physical_position_2][0] - services[i][0] > 0 and \
                            hosts_utilization_for_creature[physical_position_2][1] - services[i][1] > 0 and \
                            hosts_utilization_for_creature[physical_position_2][2] - services[i][2] > 0:
                        creature[i] = physical_position_2
                        hosts_utilization_for_creature[physical_position_2][0] = \
                            hosts_utilization_for_creature[physical_position_2][0] - services[i][0]
                        hosts_utilization_for_creature[physical_position_2][1] = \
                            hosts_utilization_for_creature[physical_position_2][1] - services[i][1]
                        hosts_utilization_for_creature[physical_position_2][2] = \
                            hosts_utilization_for_creature[physical_position_2][2] - services[j][2]
                        break

    return creature, hosts_utilization_for_creature


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
                        if dominance == -1:
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
    if solution[b][0] >= solution[a][0] and solution[b][1] >= solution[a][1] and solution[b][2] >= solution[a][2] and \
            solution[b][3] <= solution[a][3] and solution[b][4] < solution[a][4] or \
            solution[b][0] >= solution[a][0] and solution[b][1] >= solution[a][1] and solution[b][2] >= solution[a][
        2] and solution[b][3] < solution[a][3] and solution[b][4] <= solution[a][4] or \
            solution[b][0] >= solution[a][0] and solution[b][1] >= solution[a][1] and solution[b][2] > solution[a][
        2] and solution[b][3] <= solution[a][3] and solution[b][4] <= solution[a][4] or \
            solution[b][0] >= solution[a][0] and solution[b][1] > solution[a][1] and solution[b][2] >= solution[a][
        2] and solution[b][3] <= solution[a][3] and solution[b][4] <= solution[a][4] or \
            solution[b][0] > solution[a][0] and solution[b][1] >= solution[a][1] and solution[b][2] >= solution[a][
        2] and solution[b][3] <= solution[a][3] and solution[b][4] <= solution[a][4]:
        return -1
    return 0


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
    if pareto_element in pareto_head:
        pareto_head.remove(pareto_element)
    pareto_head.append(pareto_element)  # append to the beginning of the list
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
            actual_parent = posible_parent;
    return actual_parent;


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
        for j in range(h_size):
            utilization[i][j][0] = hosts[j][0] - requirements[j][0]
            utilization[i][j][1] = hosts[j][1] - requirements[j][1]
            utilization[i][j][2] = hosts[j][2] - requirements[j][2]
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
                        objectives_functions_PQ[iterator][3] != 0 or objectives_functions_PQ[iterator][4] != 0:
                    objectives_functions_P[iterator_P][0] = objectives_functions_PQ[iterator][0]
                    objectives_functions_P[iterator_P][1] = objectives_functions_PQ[iterator][1]
                    objectives_functions_P[iterator_P][2] = objectives_functions_PQ[iterator][2]
                    objectives_functions_P[iterator_P][3] = objectives_functions_PQ[iterator][3]
                    objectives_functions_P[iterator_P][4] = objectives_functions_PQ[iterator][4]
                    fronts_P[iterator_P] = fronts_PQ[iterator]
                    P[iterator_P] = PQ[iterator]
                    iterator_P += 1
    return P


def memetic_algorithm(num_creatures, NUM_GENERATIONS, services, hosts, MAX_PRIORITY):
    num_services = len(services)
    num_hosts = len(hosts)
    num_objective_functions = 5
    SELECTION_PERCENT = 0.5

    population_shape = (num_creatures, num_services)
    # try experiment:
    # 100x40 and 50x20 input sizes are considered, where AxB means A containers to B hosts.
    print(population_shape)

    P = initialize(num_creatures, num_services, num_hosts)

    # repair
    # FIXES: initializaed repaired_population, passed P instead of population to function check_feasibility_and_repair
    repaired_population = np.zeros((num_creatures, num_services))
    hosts_utilization_for_each_creature = np.zeros(
        (num_creatures, num_hosts, 3))  # 3 because we have 3 kinds of resources
    for i in range(num_creatures):
        result = check_feasibility_and_repair(services, P[i], hosts, MAX_PRIORITY)
        repaired_population[i] = result[0]
        hosts_utilization_for_each_creature[i] = result[1]

    # print("utilization from load_utilization function", load_utilization(repaired_population, hosts, services, num_creatures, num_hosts, num_services))

    # apply local search to solutions
    # FIXES use repaired_population instead of P
    P, hosts_utilization_for_each_creature = local_search(repaired_population, hosts_utilization_for_each_creature,
                                                          hosts,
                                                          services, num_creatures, num_hosts, num_services)

    # print P and hosts_utilization to verify the result
    # print("population after local search", P)
    # print("hosts_utilization", hosts_utilization_for_each_creature)
    # print("utilization from load_utilization function", load_utilization(P, hosts, services, num_creatures, num_hosts, num_services))

    # calculate the cost of each objective function for each solution
    objectives_functions_P = np.zeros((num_creatures, num_objective_functions))
    for i in range(num_creatures):
        fitness_score = fitness(P[i], services, hosts)
        objectives_functions_P[i] = fitness_score

    # calculate the non-dominated fronts
    fronts_P = non_dominated_sorting(objectives_functions_P, num_creatures)

    print("fronts", fronts_P)
    # Update set of nondominated solutions
    pareto_head = []
    for i in range(num_creatures):
        if fronts_P[i] == 1:
            pareto_head = pareto_insert(pareto_head, P[i], objectives_functions_P[i])

    print("P", P)
    print("pareto head", pareto_head)

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
        #     print("hi father!",father)
        #     print("hi mother!",mother)

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

        Q, utilization_Q = local_search(Q, utilization_Q, hosts, services, num_creatures, num_hosts, num_services)
        print("Q", Q)
        #     print("Q utilization", utilization_Q)
        #     print("Q hosts", hosts)
        # calculate the cost of each objective function for each solution
        for i in range(num_creatures):
            fitness_score = fitness(Q[i], services, hosts)
            objectives_functions_Q[i] = fitness_score
        # calculate the non-dominated fronts
        fronts_Q = non_dominated_sorting(objectives_functions_Q, num_creatures)
        # Update set of nondominated solutions Pc from Qt
        for i in range(num_creatures):
            if fronts_Q[i] == 1:
                pareto_head = pareto_insert(pareto_head, Q[i], objectives_functions_Q[i])
        #     Pt = fitness selection from Pt ∪ Qt
        print("P before evolution", P)
        print("fronts P before evolution", fronts_P)
        P = population_evolution(P, Q, objectives_functions_P, objectives_functions_Q, fronts_P, num_creatures,
                                 num_services);
    print("final P", P)
    # print("final hosts", hosts)
    # print("final services", services)
    print("final utilization", load_utilization(P, hosts, services, num_creatures, num_hosts, num_services))

    # report the best
    objectives_functions_P = np.zeros((num_creatures, num_objective_functions))
    for i in range(num_creatures):
        fitness_score = fitness(P[i], services, hosts)
        objectives_functions_P[i] = fitness_score

    fronts_best_P = non_dominated_sorting(objectives_functions_P, num_creatures)

    # MA sorting doesn't work well, temporal solution sort by best fitness value:
    cost_solution = []
    for i in range(num_creatures):
        # only the first pareto front
        if fronts_best_P[i] == 1:
            solution = P[i]
            obj_f = objectives_functions_P[i]
            cost = obj_f[0] + obj_f[1] + obj_f[2] - obj_f[3] - obj_f[4]
            cost_solution.append((solution, cost))

    cost_solution = sorted(cost_solution, key=lambda x: x[1], reverse=True)
    # print(cost_solution[0][0])
    print("all", cost_solution)
    print("best", cost_solution[0])
    return cost_solution[0][0]


def test_memetic():
    # test memetic
    print("Test memetic")
    # initialize services
    # service | CPU | MEM | DISK | PRIORITY
    services = np.array([[0.12, 0.2, 0.2, 0],
                         [0.24, 0.23, 0.1, 1],
                         [0.18, 0.20, 0.32, 0], ])

    print(services)

    # initialize hosts
    # host           | CPU | MEM | DISK | TIME | DISTANCE
    hosts = np.array([[0.5, 0.24, 0.4, 2000, 500],
                      [0.25, 0.4, 0.4, 500, 50],
                      [0.25, 0.26, 0.2, 300, 40],
                      [0.8, 0.24, 0.38, 1000, 25],
                      [0.2, 0.76, 0.62, 70, 50],
                      [0.6, 0.6, 0.5, 60, 1000],
                      [1, 1, 1, 65, 20]])

    # hosts = np.array([[0.5, 0.24, 0.4, 2000, 500],
    #                   [0.25, 0.4, 0.4, 500, 50],
    #                   [0.25, 0.26, 0.2, 300, 40],
    #                   [0.8, 0.24, 0.38, 1000, 25],
    #                   [0.2, 0.76, 0.62, 70, 50]])

    # introduce some hosts that are not currently running. For now just a copy of available hosts
    other_hosts = np.array([[0.5, 0.24, 0.4, 2000, 500],
                            [0.25, 0.4, 0.4, 500, 50],
                            [0.25, 0.26, 0.2, 300, 40],
                            [1, 1, 1, 65, 20]])

    # all together
    # constants here
    NUM_GENERATIONS = 100
    num_creatures = 10
    # num_services = 3
    # num_hosts = 5  # number of hosts may be different if we add extra during repair
    MAX_PRIORITY = 1  # max priority can be 0 or 1
    placement = memetic_algorithm(num_creatures, NUM_GENERATIONS, services, hosts, MAX_PRIORITY)
    print("Best placement: ", placement)

# test_memetic()
