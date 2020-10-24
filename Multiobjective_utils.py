import numpy
import numpy as np
import itertools
import random
MINIMIZATION = True

POSITIVE_INFINITY = float("inf")

# def archiveController(X, archive,len_archive):
#     dominated = list()
#     for ele in archive:
#         if XdominatesY(ele, X):
#             return False
#         if XdominatesY(X, ele):
#             dominated.append(ele)
#
#
#     if len(dominated) > 0:
#         for ele in dominated:
#             archive.remove(ele)
#
#     if len(archive) <= len_archive:
#         archive.append(X.copy())
#     else:
class Dominance(object):
    """Compares two solutions for dominance."""

    def __init__(self):
        super(Dominance, self).__init__()

    def __call__(self, solution1, solution2):
        return self.compare(solution1, solution2)

    def compare(self, solution1, solution2):
        """Compare two solutions.

        Returns -1 if the first solution dominates the second, 1 if the
        second solution dominates the first, or 0 if the two solutions are
        mutually non-dominated.

        Parameters
        ----------
        solution1 : Solution
            The first solution.
        solution2 : Solution
            The second solution.
        """
        raise NotImplementedError("method not implemented")

class ParetoDominance(Dominance):
    """Pareto dominance with constraints.

    If either solution violates constraints, then the solution with a smaller
    constraint violation is preferred.  If both solutions are feasible, then
    Pareto dominance is used to select the preferred solution.
    """

    def __init__(self):
        super(ParetoDominance, self).__init__()

    def compare(self, solution1, solution2):
        dominate1 = False
        dominate2 = False

        for i in range(len(solution1.fitness)):
            o1 = solution1.fitness[i]
            o2 = solution2.fitness[i]

            if MINIMIZATION:
                if o1 < o2:
                    dominate1 = True

                    if dominate2:
                        return 0
                elif o1 > o2:
                    dominate2 = True

                    if dominate1:
                        return 0

        if dominate1 == dominate2:
            return 0
        elif dominate1:
            return -1
        else:
            return 1


class Archive(object):
    """An archive only containing non-dominated solutions."""

    def __init__(self, dominance=ParetoDominance()):
        super(Archive, self).__init__()
        self._dominance = dominance
        self._contents = []

    def add(self, solution):
        flags = [self._dominance.compare(solution, s) for s in self._contents]
        dominates = [x > 0 for x in flags]
        nondominated = [x == 0 for x in flags]

        if any(dominates):
            return False
        else:
            self._contents = list(itertools.compress(self._contents, nondominated)) + [solution]
            return True

    def append(self, solution):
        self.add(solution)

    def extend(self, solutions):
        for solution in solutions:
            self.append(solution)

    def remove(self, solution):
        try:
            self._contents.remove(solution)
            return True
        except ValueError:
            return False

    def __len__(self):
        return len(self._contents)

    def __getitem__(self, key):
        return self._contents[key]

    def __iadd__(self, other):
        if hasattr(other, "__iter__"):
            for o in other:
                self.add(o)
        else:
            self.add(other)

        return self

    def __iter__(self):
        return iter(self._contents)

class Archive(object):
    """An archive only containing non-dominated solutions."""

    def __init__(self, dominance=ParetoDominance()):
        super(Archive, self).__init__()
        self._dominance = dominance
        self._contents = []

    def add(self, solution):
        flags = [self._dominance.compare(solution, s) for s in self._contents]
        dominates = [x > 0 for x in flags]
        nondominated = [x == 0 for x in flags]
        print(dominates,nondominated)
        if any(dominates):
            return False
        else:
            self._contents = list(itertools.compress(self._contents, nondominated)) + [solution]
            return True

    def append(self, solution):
        self.add(solution)

    def extend(self, solutions):
        for solution in solutions:
            self.append(solution)

    def remove(self, solution):
        try:
            self._contents.remove(solution)
            return True
        except ValueError:
            return False

    def __len__(self):
        return len(self._contents)

    def __getitem__(self, key):
        return self._contents[key]

    def __iadd__(self, other):
        if hasattr(other, "__iter__"):
            for o in other:
                self.add(o)
        else:
            self.add(other)

        return self

    def __iter__(self):
        return iter(self._contents)

class AdaptiveGridArchive(Archive):

    def __init__(self, capacity, nobjs, divisions, dominance=ParetoDominance()):
        super(AdaptiveGridArchive, self).__init__(dominance)
        self.capacity = capacity
        self.nobjs = nobjs
        self.divisions = divisions

        self.adapt_grid()

    def add(self, solution):
        # check if the candidate solution dominates or is dominated
        flags = [self._dominance.compare(solution, s) for s in self._contents]
        dominates = [x > 0 for x in flags]
        nondominated = [x == 0 for x in flags]

        #dominated by one residence dont let it enter
        if any(dominates):
            return False
        #remove the dominatedresident

        self._contents = list(itertools.compress(self._contents, nondominated))

        # archive is empty, add the candidate
        if len(self) == 0:
            self._contents.append(solution)
            self.adapt_grid()
            return True

        # temporarily add the candidate solution
        self._contents.append(solution)
        index = self.find_index(solution)

        if index < 0:
            self.adapt_grid()
            index = self.find_index(solution)
        else:
            self.density[index] += 1

        if len(self) <= self.capacity:
            # keep the candidate if size is less than capacity
            return True
        elif self.density[index] == self.density[self.find_densest()]:
            # reject candidate if in most dense cell
            self.remove(solution)
            return False
        else:
            # keep candidate and remove one from densest cell
            self.remove(self.pick_from_densest())
            return True

    def remove(self, solution):
        removed = super(AdaptiveGridArchive, self).remove(solution)

        if removed:
            index = self.find_index(solution)

            if self.density[index] > 1:
                self.density[index] -= 1
            else:
                self.adapt_grid()

        return removed

    def adapt_grid(self):
        self.minimum = [POSITIVE_INFINITY] * self.nobjs
        self.maximum = [-POSITIVE_INFINITY] * self.nobjs
        self.density = [0.0] * (self.divisions ** self.nobjs)

        for solution in self:
            for i in range(self.nobjs):
                self.minimum[i] = min(self.minimum[i], solution.fitness[i])
                self.maximum[i] = max(self.maximum[i], solution.fitness[i])

        for solution in self:
            self.density[self.find_index(solution)] += 1

    def find_index(self, solution):
        index = 0

        for i in range(self.nobjs):
            value = solution.fitness[i]

            if value < self.minimum[i] or value > self.maximum[i]:
                return -1

            if self.maximum[i] > self.minimum[i]:
                value = (value - self.minimum[i]) / (self.maximum[i] - self.minimum[i])
            else:
                value = 0

            temp_index = int(self.divisions * value)

            if temp_index == self.divisions:
                temp_index -= 1

            index += temp_index * pow(self.divisions, i)

        return index

    def find_densest(self):
        index = -1
        value = -1

        for i in range(len(self)):
            temp_index = self.find_index(self[i])
            temp_value = self.density[temp_index]

            if temp_value > value:
                value = temp_value
                index = temp_index

        return index

    def pick_from_densest(self):
        solution = None
        value = -1

        for i in range(len(self)):
            temp_value = self.density[self.find_index(self[i])]

            if temp_value > value:
                solution = self[i]
                value = temp_value

        return solution

    def pick_from_sparse(self,beta=-4):
        temp_list = list()
        # print(self.density)

        for i in range(len(self)):
            candidate_density = self.density[self.find_index(self[i])]
            if (candidate_density == 0):
                self.adapt_grid()
                # print(self.find_index(self[i]))
                candidate_density = self.density[self.find_index(self[i])]
                # print(temp_value)


            temp_value = pow(candidate_density,beta)

            temp_list.append(temp_value)
        prob_list = [x/sum(temp_list) for x in temp_list]
        c = np.cumsum(prob_list)
        r = np.random.rand()
        for i in range(len(c)):
            if r <= c[i]:
                idx = i
                break

        return self[i]






def RouletteWheelSelection(p):

    r=np.random.random()
    c=np.cumsum(p)
    i=next(x[0] for x in enumerate(c) if x[1] >= r)

    return i


def find(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def GetOccupiedCells(pop):
    GridIndices = []
    for i in range(len(pop)):
        GridIndices.append(pop[i].GridIndex)
    try:
        occ_cell_index = list(set(GridIndices))
    except:
        print(GridIndices)

    occ_cell_member_count = list(np.zeros(shape=len(occ_cell_index)))

    for k in range(len(occ_cell_index)):
        occ_cell_member_count[k] = GridIndices.count(occ_cell_index[k])

    return occ_cell_index, occ_cell_member_count


def SelectLeader(rep, beta):


    occ_cell_index, occ_cell_member_count = GetOccupiedCells(rep)
    p = [x ** (-beta) for x in occ_cell_member_count]
    p = [x / sum(p) for x in p]

    selected_cell_index = occ_cell_index[RouletteWheelSelection(p)]

    GridIndices = []
    for i in range(len(rep)):
        GridIndices.append(rep[i].GridIndex)

    selected_cell_members = find(GridIndices, lambda x: x == selected_cell_index)
    n = len(selected_cell_members)

    selected_member_index = round(random.uniform(1, n))

    h = selected_cell_members[selected_member_index - 1]

    rep_h = rep[h]

    return rep_h



def Dominates(x,y):

    try:
        x=x.fitness
    except AttributeError:
        pass
    try:
        y=y.fitness
    except AttributeError:
        pass
    dom= (all(i<=j for i, j in zip(x, y)) & any(i<j for i, j in zip(x, y)))

    return dom


def DetermineDomination(pop):
    npop=len(pop)
    for i in range(len(pop)):
        pop[i].Dominated=False
        for j in range(0,i-1):
            if not pop[j].Dominated:
                if Dominates(pop[i],pop[j]):
                    pop[j].Dominated=True
                elif Dominates(pop[j],pop[i]):
                    pop[i].Dominated=True
                    break;
    return pop



def EliminateDuplicates(rep):
    total=[]
    rep2 = []
    for i in range(len(rep)):
        if list(rep[i].pos) not in [list(item) for item in total]:
            rep2.append(rep[i])
        total.append(rep[i].pos)
    return rep2


def GetNonDominatedParticles(pop):
    nd_pop = []
    for i in range(len(pop)):
        if not pop[i].Dominated:
            nd_pop.append(pop[i])

    nd_pop = EliminateDuplicates(nd_pop)

    return nd_pop


def GetCosts(A):
    costs = []
    for i in range(len(A)):
        costs.append(A[i].fitness)

    return costs


def Cost_normalize(item, costs):
    if max(costs) != min(costs):
        item_norm = (item - min(costs)) / (max(costs) - min(costs))
    else:
        item_norm = (item - min(costs)) / (np.mean(costs))
    # if np.isnan(item_norm):
        # print()

    return item_norm


def CreateHypercubes(costs, ngrid, alpha):
    nobj = 2
    G = [[[], []], [[], []]]

    cost1 = [item[0] for item in costs]
    cost1_norm = [Cost_normalize(item, cost1) for item in cost1]

    cost2 = [item[1] for item in costs]
    cost2_norm = [Cost_normalize(item, cost2) for item in cost2]

    costs = list(zip(cost1_norm, cost2_norm))

    for j in range(nobj):
        min_cj = min([i[j] for i in costs])
        max_cj = max([i[j] for i in costs])

        dcj = alpha * (max_cj - min_cj)

        min_cj = min_cj - dcj
        max_cj = max_cj + dcj

        gx = list(np.linspace(min_cj, max_cj, ngrid - 1))

        G[j][0] = [-np.inf, *gx]
        G[j][1] = [*gx, np.inf]

    return G

def sub2ind(array_shape, rows, cols):
    return cols*array_shape[1] + rows+1


def GetGridIndex(particle, G, pop):
    c = particle.fitness
    # if np.isnan(c[0]) or np.isnan(c[1]):
        # print()
    costs = GetCosts(pop)

    c = (Cost_normalize(c[0], [item[0] for item in costs]), Cost_normalize(c[1], [item[1] for item in costs]))

    nobj = 2
    ngrid = len(G[0][0])
    SubIndex = [0, 0]
    # print(c)
    # print (x[0] for x in enumerate(G[1][1]) if x[1] > c[j])
    for j in range(nobj):
        U = G[j][1]
        try:
            i = next(x[0] for x in enumerate(U) if x[1] > c[j])
        except:
            print(U)
            print(costs)
            raise ValueError
        SubIndex[j] = i

    Index = sub2ind([ngrid, ngrid], SubIndex[0], SubIndex[1])

    return Index, SubIndex


def DeleteFromRep(rep, EXTRA, gamma, GreyWolves, GreyWolves_num):
    if rep == []:
        rep.append(GreyWolves[round(random.uniform(0, GreyWolves_num)) - 1])
        gamma = 1

    for k in range(EXTRA):

        occ_cell_index, occ_cell_member_count = GetOccupiedCells(rep)
        p = [x ** (-gamma) for x in occ_cell_member_count]
        p = [x / sum(p) for x in p]

        selected_cell_index = occ_cell_index[RouletteWheelSelection(p)]

        GridIndices = []
        for i in range(len(rep)):
            GridIndices.append(rep[i].GridIndex)

        selected_cell_members = find(GridIndices, lambda x: x == selected_cell_index)
        n = len(selected_cell_members)

        selected_member_index = round(random.uniform(1, n))

        j = selected_cell_members[selected_member_index - 1]

        rep.pop(j)

    return rep

# if __name__ == '__main__':
#     archive = AdaptiveGridArchive(capacity=5,nobjs=2,divisions=10)
#     archive += ['1','2','3']