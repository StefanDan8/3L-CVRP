import numpy as np
import matplotlib.pyplot as plt
import time
from tsp_reader import TSPProblem


class ACO:
    """
    Ant Colony Optimization
    """

    def __init__(self, alpha = 1, beta = 2, rho = 0.5, nn = 20):
        self.alpha = alpha  # pheromone exponent
        self.beta = beta  # prior exponent
        self.rho = rho  # evaporation factor
        self.nn = nn  # nearest-neighbour threshold
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
        self.history_best_length = []

    def __call__(self, TSP_instance: TSPProblem):
        self.distance_matrix = TSP_instance.distance_matrix
        self.num_cities = TSP_instance.DIMENSION
        self.num_ants = self.num_cities  # the number of ants. Usually it is equal to the number of cities
        self.nn_list = TSP_instance.nn_list
        self.ants = [SingleAnt(self.num_cities) for _ in range(self.num_ants)]
        self.prior = np.power(self.distance_matrix, -self.beta)

    def _compute_tour_length(self, ant):
        tour_length = 0.0
        for i in range(self.num_cities):
            tour_length += self.distance_matrix[ant.tour[i], ant.tour[i + 1]]
        return tour_length


class SingleAnt:
    def __init__(self, num_cities):
        self.num_cities = num_cities
        self.tour_length = 0.0  # ant's tour length
        self.tour = np.zeros(shape = self.num_cities + 1, dtype = 'int')  # store the (partial) tour
        self.visited = [False for _ in range(self.num_cities)]  # visited cities


class AS(ACO):
    """
    Ant System
    """

    def __init__(self, alpha = 1, beta = 5, rho = 0.5, nn = 20, seed = 42):
        super().__init__(alpha, beta, rho, nn)
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
            start = time.time()
            # Construct Solutions
            self._construct_solutions()
            # Optional Local Search Improvement
            # self._local_search()  #TODO
            # Update Pheromones
            self._update_pheromones()

            # Statistics & Performance:
            print(f'Iteration {self.current_iter}')
            print(f'Time: {np.round(time.time()-start, 2)}')
            print(f'Best tour length: {self.best_tour_length}')

        print(f'Absolute time: {time.time()-absolute_start}')
        return self.best_tour, self.best_tour_length

    def _construct_solutions(self):
        """
        Sequentially constructs a tour for each ant based on the `_decision_rule`
        """
        # TODO implement this in parallel
        step = 0
        for ant in self.ants:
            # Empty ants' memory
            ant.visited = [False for _ in range(ant.num_cities)]
            # Assign each ant a random starting city
            rand_int = self.rng.integers(low = 0, high = self.num_cities)
            ant.tour[step] = rand_int
            ant.visited[rand_int] = True
        # Tour construction
        while step < self.num_cities - 1:
            for ant in self.ants:
                self._decision_rule(ant, step)
            step += 1
        # close the tour
        for ant in self.ants:
            ant.tour[self.num_cities] = ant.tour[0]  # unnecessary, but makes it clearer
            ant.tour_length = self._compute_tour_length(ant)
            # Update best tour so far
            if ant.tour_length < self.best_tour_length:
                self.best_tour_length = ant.tour_length
                self.best_tour = ant.tour

        # Add the best tour length to history after this iteration
        self.history_best_length.append(self.best_tour_length)

    def _decision_rule(self, ant: SingleAnt, step: int):
        """
        Randomly picks next city for the `ant` based on a distribution derived from the current city's row in
        `self.choice_info`
        :param ant: the Ant agent
        :param step: current step
        """
        c = ant.tour[step]  # current city
        prob = np.array([self.choice_info[c, j] if not ant.visited[j] else 0.0 for j in range(self.num_cities)])
        prob = prob / sum(prob)  # construct a probability distribution based on edge attractiveness

        next_city = self.rng.choice(self.num_cities, size = 1, p = prob).item()
        ant.tour[step + 1] = next_city
        ant.visited[next_city] = True

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

    def _local_search(self):
        pass

    def _choose_next_best(self, ant: SingleAnt, step: int):
        pass

    def _update_pheromones(self):
        # Evaporate
        self.pheromone = (1 - self.rho) * self.pheromone
        # Deposit pheromone
        for ant in self.ants:
            self._deposit_pheromone(ant)
        # Update choice_info
        self._compute_choice_info()

    def _deposit_pheromone(self, ant: SingleAnt):
        """
        Update the pheromone matrix, `self.pheromone`, by adding a quantity which is inversely proportional to the
        length of the tour made by `ant` to each value part of its tour
        :param ant: the Ant agent
        """
        delta = 100 / ant.tour_length
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
