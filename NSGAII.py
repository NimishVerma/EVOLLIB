import numpy as np
import pygmo as pg
import math
import pygmo as pg
import random as random
from copy import deepcopy
from collections import defaultdict
import collections
collections.abc.Sequence.register(np.ndarray)
#Function to sort by values
def sort_by_values(list1, values):
    tmp_values = deepcopy(values)
    sorted_list = []
    hit=0
    while(len(sorted_list)!=len(list1)):
        minimal = min(tmp_values)

        # print(np.where(tmp_values == minimal))
        try:
            idx = np.where(tmp_values == minimal)[0][0]
        except:
            if minimal == 4444444444444444:
                if hit ==0:
                    idx = 0
                    hit+= 1
                elif hit ==1:
                    idx=len(list1)-1
                    hit+=1
                else:
                    print("HIT>2")
        if idx in list1:
            sorted_list.append(idx)
        tmp_values[idx] = math.inf
    return sorted_list


def dominates(obj1, obj2):
    """Return true if each objective of *self* is not strictly worse than
            the corresponding objective of *other* and at least one objective is
            strictly better.
        **no need to care about the equal cases
        (Cuz equal cases mean they are non-dominators)
    :param obj1: a list of multiple objective values
    :type obj1: numpy.ndarray
    :param obj2: a list of multiple objective values
    :type obj2: numpy.ndarray
    :param sign: target types. positive means maximize and otherwise minimize.
    :type sign: list
    """
    indicator = False
    for a, b in zip(obj1, obj2):
        if a < b:
            indicator = True
        # if one of the objectives is dominated, then return False
        elif a > b:
            return False
    return indicator


def sortNondominated(fitness, k=None, first_front_only=False):
    """Sort the first *k* *individuals* into different nondomination levels
        using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
        see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
        where :math:`M` is the number of objectives and :math:`N` the number of
        individuals.
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param first_front_only: If :obj:`True` sort only the first front and
                                    exit.
        :param sign: indicate the objectives are maximized or minimized
        :returns: A list of Pareto fronts (lists), the first list includes
                    nondominated individuals.
        .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
            non-dominated sorting genetic algorithm for multi-objective
            optimization: NSGA-II", 2002.
    """
    if k is None:
        k = len(fitness)

    # Use objectives as keys to make python dictionary
    map_fit_ind = defaultdict(list)
    for i, f_value in enumerate(fitness):  # fitness = [(1, 2), (2, 2), (3, 1), (1, 4), (1, 1)...]
        map_fit_ind[f_value].append(i)
    fits = list(map_fit_ind.keys())  # fitness values

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)  # n (The number of people dominate you)
    dominated_fits = defaultdict(list)  # Sp (The people you dominate)

    # Rank first Pareto front
    # *fits* is a iterable list of chromosomes. Each has multiple objectives.
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i + 1:]:
            # Eventhougn equals or empty list, n & Sp won't be affected
            if dominates(fit_i, fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif dominates(fit_j, fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]  # The first front
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all individuals are sorted or
    # the given number of individual are sorted.
    # If Sn=0 then the set of objectives belongs to the next front
    if not first_front_only:  # first front only
        N = min(len(fitness), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                # Iterate Sn in current fronts
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1  # Next front -> Sn - 1
                    if dominating_fits[fit_d] == 0:  # Sn=0 -> next front
                        next_front.append(fit_d)
                         # Count and append chromosomes with same objectives
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        #get solutions that dominate p
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            #get solutions that p dominates
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[-1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values1[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values2[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

def oldmutate(individual,lb,ub):
    if random.random()<0.05:
        rand_dim = random.randint(1,len(individual)-1)
        individual[rand_dim] = lb[rand_dim] + (ub[rand_dim]-lb[rand_dim])*random.random()
    individual=np.clip(individual,lb,ub)
    return individual

def oldcrossover(a,b,lb,ub):
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    if random.random()<0.5:
        individual = oldmutate((a+b)/2,lb,ub)
    else:
        individual = oldmutate((a+b)/2,lb,ub)
    individual=np.clip(individual, lb, ub)
    return individual

#Function to carry out the crossover SPX
def crossover(a,b,lb,ub,eta=0.2):
    # :param     # eta: Crowding     # degree      # of     # the    # crossover.A    # high    # eta   # will    # produce    # children    # resembling    # to    # their    # parents,
    # while a small eta will produce solutions much more different.
    r=random.random()
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    child1, child2 = np.zeros((len(a))),np.zeros((len(a)))
    for i, (x1, x2) in enumerate(zip(a, b)):
        rand = random.random()
        if rand <= 0.5:
            beta = 2. * rand
        else:
            beta = 1. / (2. * (1. - rand))
        beta **= 1. / (eta + 1.)
        child1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
        child2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))

    return oldmutate(child1,lb,ub),oldmutate(child2,lb,ub)

def mutPolynomialBounded(individual, low, up,eta=10):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param low: A value or a :term:`python:sequence` of values that
                is the lower bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that
               is the upper bound of the search space.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    for i, xl, xu in zip(range(size), low, up):

        x = individual[i]
        delta_1 = (x - xl) / (xu - xl)
        delta_2 = (xu - x) / (xu - xl)
        rand = random.random()
        mut_pow = 1.0 / (eta + 1.)

        if rand < 0.5:
            xy = 1.0 - delta_1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
            delta_q = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta_2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
            delta_q = 1.0 - val ** mut_pow

        x = x + delta_q * (xu - xl)
        x = min(max(x, xl), xu)
        individual[i] = x
    return individual
#Function to find index of list

def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1
#Main program starts here

def tournament(solution,function1_values,function2_values,crowding_distance_values):
    participants = random.sample(solution, 2)
    idx0,idx1 = -1, -1
    for i,ele in enumerate(solution):
        if all(ele == participants[0]):
            idx0=i
        elif all(ele==participants[1]):
            idx1=i
        if idx0 > -1 and idx1 >-1:
            break



    obj1 = [function1_values[idx0], function2_values[idx0]]
    obj2 = [function1_values[idx1], function2_values[idx1]]

    if dominates(np.array(obj1),np.array(obj2)):
        a1 = idx0
    # elif crowding_distance_values[idx0]>crowding_distance_values[idx1]:
    #     a1 = participants[0]
    else:
        a1 = idx1
    return a1

def NSGAII(objf,lb,ub,dim,PopSize,iters):
    crossover_prob = 0.9
    mutation_prob = 0.05
    pop_size = PopSize
    max_gen = iters
    prob = objf
    #Initialization
    min_x=lb
    max_x=ub
    solution = np.zeros((PopSize, dim))

    for i in range(dim):
        solution[:,i] = np.random.uniform(0,1, PopSize) * (ub[i] - lb[i]) + lb[i]
    gen_no=0
    while(gen_no<max_gen):
        # print(gen_no)
        if not isinstance(solution, np.ndarray):
            solution = np.array(solution)
        function1_values,function2_values = np.zeros((len(solution))), np.zeros((len(solution)))
        for i in range(0, pop_size):
            # solution[i] = np.clip(solution[i], lb, ub)
            fitness_val = prob.fitness(solution[i,:])
            function1_values[i]=fitness_val[0]
            function2_values[i]=fitness_val[1]

        non_dominated_sorted_solution = sortNondominated(list(zip(function1_values[:],function2_values[:])))

        # print("The best front for Generation number ",gen_no, " is")
        #changed this to -1 from 0, since -1 has best front
        # for valuez in non_dominated_sorted_solution[-1]:
        #     print(solution[valuez],end=" ")
        # print("\n")
        crowding_distance_values=[]
        for i in range(0,len(non_dominated_sorted_solution)):
            crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
        solution2 = solution[:]
        #Generating offsprings
        while(len(solution2)!=2*pop_size):
            # print(solution2)
            a1 = tournament(solution,function1_values,function2_values,crowding_distance_values)
            b1 = tournament(solution,function1_values,function2_values,crowding_distance_values)
            while b1 == a1:
                b1 = tournament(solution, function1_values, function2_values, crowding_distance_values)
            if random.random() < 1:#crossover_prob:
                solution2 = np.vstack((solution2, crossover(deepcopy(solution[a1]),deepcopy(solution[b1]), lb, ub)))

            if len(solution2) == 2 * pop_size:
                break
            if len(solution2) -2 * pop_size>0:
                solution2=solution2[:2*pop_size]

            # for child_to_mutate in solution2:
            #     if len(solution2) == 2 * pop_size:
            #         break
            #     if random.random() < mutation_prob:
            #         solution2 = np.vstack((solution2, mutPolynomialBounded(deepcopy(child_to_mutate), lb, ub)))
        function1_values2, function2_values2 = np.zeros((len(solution2))), np.zeros((len(solution2)))
        for i in range(0, 2 * pop_size):
            solution2[i] = np.clip(solution2[i], lb, ub)
            fitness_val = prob.fitness(solution2[i,:])
            function1_values2[i]=fitness_val[0]
            function2_values2[i]=fitness_val[1]
        non_dominated_sorted_solution2 = sortNondominated(list(zip(function1_values2[:],function2_values2[:])))
        crowding_distance_values2=[]
        for i in range(0,len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
        new_solution= []
        for i in range(len(non_dominated_sorted_solution2)):#-1,-1,-1):
            if len(new_solution)+len(non_dominated_sorted_solution2[i])<=pop_size:
                new_solution.extend(non_dominated_sorted_solution2[i])
                if (len(new_solution) == pop_size):
                    break
            else:
                non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
                front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])

                front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
                front.reverse()
                for value in front:
                    new_solution.append(value)
                    if(len(new_solution)==pop_size):
                        break
                if (len(new_solution) == pop_size):
                    break
        solution = [solution2[i] for i in new_solution]
        gen_no = gen_no + 1

    #Lets plot the final front now
    # function1 = [i * 1 for i in function1_values]
    # function2 = [j * -1 for j in function2_values]
    final_f1, final_f2 = [], []
    for i in non_dominated_sorted_solution[0]:
        final_f1.append(function1_values[i])
        final_f2.append(function2_values[i])
    return np.column_stack((final_f1,final_f2))

import time

# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return costs[is_efficient]

def main(problem_id, file_name):
    prob1 = pg.problem(pg.cec2009(problem_id))
    lb, ub = prob1.get_bounds()
    lb, ub = lb.tolist(), ub.tolist()
    # print(lb,ub)
    timerStart = time.time()
    Archive_costs = NSGAII(prob1, lb, ub, 30,100,500)#(objf,lb,ub,dim,PopSize,iters)
    timerEnd = time.time()
    print("Execution time=",timerEnd-timerStart)

    # print("Length of archive: ", len( returned_archive))
    # print(returned_archive)

    import numpy as np
    import csv
    import matplotlib.pyplot as plt
    with open('C:\\Users\\nimis\\Desktop\\pds jpgs xls docx\\EvoloPy-master\\MO Result\\'+file_name + ".csv", 'a', newline='\n') as out:
        writer = csv.writer(out, delimiter=',')
        for row in np.asarray(Archive_costs):
            writer.writerow(row)
    out.close()
    # pareto_front = np.asarray([x.fitness for x in returned_archive._contents])
    # print(pareto_front)
    x, y = np.asarray(Archive_costs).T

    if problem_id == 6:
        X = np.linspace(1 / 4, 1 / 2, 20)
        X2 = np.linspace(3 / 4, 1, 20)

        X_final = np.concatenate(([0], X, X2))
        Y_final = 1 - X_final
    elif problem_id == 7:
        X_final = np.linspace(0, 1, 100, True)
        Y_final = 1 - X_final
    elif problem_id == 5:
        X_final = np.zeros(20)
        for i in range(20):
            X_final[i] = i / 20
        Y_final = 1 - X_final
    elif problem_id == 4:
        X_final = np.linspace(0, 1, 100, True)
        Y_final = 1 - X_final ** 2
    elif problem_id in [3, 2, 1]:
        X_final = np.linspace(0, 1, 100, True)
        Y_final = 1 - np.sqrt(X_final)
    f = plt.figure(1)
    plt.title('Multipop ' + prob1.get_name())
    plt.xlim([0, 2.5])
    plt.ylim([0, 2.5])
    plt.scatter(x, y)
    if problem_id in [6, 7]:
        plt.scatter(X_final, Y_final)
    else:
        plt.plot(X_final, Y_final)
    plt.savefig('C:\\Users\\nimis\\Desktop\\pds jpgs xls docx\\EvoloPy-master\\MO Result\\'+file_name+'.png')
    f.show()

    # print(pop)n

for problem_id in range(3,8):

    for nrun in range(30):
        if problem_id == 3 and nrun < 2:
            continue
        print("RUN {0} UF{1}".format(nrun, problem_id))
        file_name = "NSGAII UF{0}_{1}".format(problem_id,nrun)
        main(problem_id, file_name)


    print ("+++++++++++++++++++++++ {}".format(problem_id))



