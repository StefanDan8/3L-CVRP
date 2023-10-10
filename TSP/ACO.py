import matplotlib.pyplot as plt
import numpy as np
import time
from tsp_reader import TSPProblem


class ACO:
    """
    Ant Colony Optimization
    """

    def __init__(self, alpha = 1, beta = 2, rho = 0.5, Q = 20, nn = 20):
        self.alpha = alpha  # pheromone exponent
        self.beta = beta  # prior exponent
        self.rho = rho  # evaporation factor
        self.nn = nn  # nearest-neighbour threshold
        self.Q = Q  # pheromone delta multiplier
        self.num_cities = 0
        self.num_ants = 0
        self.distance_matrix = None
        self.nn_list = None  # matrix with nearest neighbor lists of depth nn -- can bring speed-up
        self.pheromone = None  # pheromone matrix
        self.choice_info = None  # combined pheromone and heuristic information
        # this stores tau^alpha * eta^beta --> in an iteration, each ant uses these values, so it makes sense to compute
        # them only once then make them available to all ants
        # to spare computation, at the cost of possibly not always finding valid tours, one can limit the size to
        # num_cities x nn  as it is unlikely that far cities are ever preferred by ants
        # in the same fashion, eta^beta can be stored somewhere as these values don't change
        self.prior = None
        self.ants = None

        # Statistics & Performance Variables
        self.max_iter = 100
        self.current_iter = 0
        self.best_tour = None
        self.best_tour_length = np.Inf
        self.evolution_best_length = []
        self.iteration_best_length = []

    def __call__(self, TSP_instance: TSPProblem):
        self.distance_matrix = TSP_instance.distance_matrix
        self.num_cities = TSP_instance.DIMENSION
        self.num_ants = self.num_cities  # the number of ants. Usually it is equal to the number of cities
        self.nn_list = TSP_instance.nn_list
        self.ants = [SingleAnt(self.num_cities) for _ in range(self.num_ants)]
        self.prior = np.power(self.distance_matrix, -self.beta)

    def _compute_tour_length(self, tour):
        tour_length = 0.0
        for i in range(self.num_cities):
            tour_length += self.distance_matrix[tour[i], tour[i + 1]]
        return tour_length


class SingleAnt:
    def __init__(self, num_cities):
        self.num_cities = num_cities
        self.tour_length = 0.0  # ant's tour length
        self.tour = np.zeros(shape = self.num_cities + 1, dtype = 'int')  # store the (partial) tour
        self.visited = [False for _ in range(self.num_cities)]  # visited cities

    def reset(self, rng):
        self.tour_length = 0.0
        self.visited = [False for _ in range(self.num_cities)]
        rand_int = rng.integers(low = 0, high = self.num_cities)
        self.tour[0] = rand_int
        self.visited[rand_int] = True

    def construct_tour(self, choice_info, rng):
        self.reset(rng)

        def _decision_rule(step: int):
            """
            Randomly picks next city for the ant based on a distribution derived from the current city's row in
            `choice_info`
            :param step: current step
            """
            c = self.tour[step - 1]  # current city
            prob = np.array([choice_info[c, j] if not self.visited[j] else 0.0 for j in range(self.num_cities)])
            prob = prob / sum(prob)  # construct a probability distribution based on edge attractiveness
            return rng.choice(self.num_cities, size = 1, p = prob).item()

        for step in range(1, self.num_cities):
            next_city = _decision_rule(step)
            self.tour[step] = next_city
            self.visited[next_city] = True
            self.tour[self.num_cities] = self.tour[0]
        return self.tour


class AS(ACO):
    """
    Ant System
    """

    def __init__(self, alpha = 1, beta = 5, rho = 0.5, Q = 20, nn = 20, seed = 42):
        super().__init__(alpha, beta, rho, Q, nn)
        self.rng = np.random.default_rng(seed)

    def __call__(self, TSP_instance: TSPProblem):
        """
        Solves an instance of Traveling Salesman Problem by using the Ant System Approach, from the family of Ant Colony
        Optimization Metaheuristic
        :param TSP_instance: instance of Traveling Salesman Problem
        :return: best tour found and its length
        """
        # Data Initialization
        super().__call__(TSP_instance)
        self.pheromone = self._initialize_pheromone_matrix(self.num_cities, TSP_instance.nn_length)
        self.choice_info = self._compute_choice_info()

        absolute_start = time.time()

        # AS Iteration
        while not self._terminate():
            self.current_iter += 1
            # start = time.time()
            # Construct Solutions
            tour, length = self._construct_solutions()
            # Optional Local Search Improvement
            improved_tour = self._local_search(tour, length)
            improved_length = self._compute_tour_length(improved_tour)
            if improved_length < self.best_tour_length:
                self.best_tour_length = improved_length
                print(improved_length)
            # Update Pheromones
            self._update_pheromones(improved_tour, improved_length)

            # Statistics & Performance:
            # print(f'Iteration {self.current_iter}')
            # print(f'Time: {np.round(time.time()-start, 2)}')
            # print(f'Best tour length: {self.best_tour_length}')

        print(f'Absolute time: {time.time() - absolute_start}')
        plt.plot(range(1, self.max_iter + 1), self.iteration_best_length)
        plt.show()
        return self.best_tour, self.best_tour_length

    def _construct_solutions(self):
        """
        Sequentially constructs a tour for each ant based on the `_decision_rule`
        """
        best_iter_tour_length = np.Inf
        best_ant = None
        for ant in self.ants:
            ant.tour_length = self._compute_tour_length(ant.construct_tour(self.choice_info, self.rng))
            if ant.tour_length < best_iter_tour_length:
                best_iter_tour_length = ant.tour_length
                best_ant = ant
        if best_iter_tour_length < self.best_tour_length:
            self.best_tour_length = best_iter_tour_length
            self.best_tour = best_ant.tour
        self.iteration_best_length.append(best_iter_tour_length)
        self.evolution_best_length.append(self.best_tour_length)
        return best_ant.tour, best_iter_tour_length

    def _nn_decision_rule(self, ant: SingleAnt, step: int):
        """
        Same as the basic `_decision_rule`, only that the search space is limited to `self.nn` closest neighbors.
        If all `self.nn` closest neighbours have already been visited, then the `_choose_next_best` function is called
        to find a choice among the cities not visited yet
        :param ant: the Ant agent
        :param step: current step
        """
        # TODO
        pass

    def _local_search(self, tour, tour_length):
        def dist(v, w):
            return tour_length - self.distance_matrix[tour[v], tour[v + 1]] - \
                   self.distance_matrix[tour[w], tour[w + 1]] + \
                   self.distance_matrix[tour[v], tour[w]] + \
                   self.distance_matrix[tour[v + 1], tour[w + 1]]

        def do2_Opt(path, v, w):
            path[v + 1:w + 1] = path[v + 1:w + 1][::-1]

        num_iter = 0
        for i in range(self.num_cities - 1):
            for j in range(i, self.num_cities):
                if dist(i, j) < tour_length:
                    # found improvement
                    num_iter += 1
                    do2_Opt(tour, i, j)
                    if num_iter == 5:
                        return tour
        return tour

    def _choose_next_best(self, ant: SingleAnt, step: int):
        pass

    def _update_pheromones(self, tour, length):
        # Evaporate
        self.pheromone = (1 - self.rho) * self.pheromone
        # Deposit pheromone
        for ant in self.ants:
            self._deposit_pheromone(ant)
        # Deposit for local search tour
        delta = self.Q / length
        for i in range(self.num_cities):
            j = tour[i]
            k = tour[i + 1]
            self.pheromone[j, k] += delta
            self.pheromone[k, j] += delta
        # Update choice_info
        self._compute_choice_info()

    def _deposit_pheromone(self, ant: SingleAnt):
        """
        Update the pheromone matrix, `self.pheromone`, by adding a quantity which is inversely proportional to the
        length of the tour made by `ant` to each value part of its tour
        :param ant: the Ant agent
        """
        delta = self.Q / ant.tour_length
        for i in range(self.num_cities):
            j = ant.tour[i]
            k = ant.tour[i + 1]
            self.pheromone[j, k] += delta
            self.pheromone[k, j] += delta

    def _initialize_pheromone_matrix(self, num_ants: int, nn_length: int):
        return np.ones(shape = (self.num_cities, self.num_cities)) * (num_ants / nn_length)

    def _compute_choice_info(self):
        return np.power(self.pheromone, self.alpha) * self.prior

    def _terminate(self):
        """
        Termination Condition
        Currently very simple: maximum number of iterations is `self.max_iter`
        :return: True if the maximum number of iterations is reached, False otherwise.
        """
        return self.current_iter == self.max_iter
