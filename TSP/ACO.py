import numpy as np

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
        self.distance_matrix = None
        self.num_cities = 0
        self.nn_list = None  # matrix with nearest neighbor lists of depth nn -- can bring speed-up
        self.pheromone = None  # pheromone matrix
        self.choice_info = None  # combined pheromone and heuristic information
        # this stores tau^alpha * eta^beta --> in an iteration, each ant uses these values, so it makes sense to compute
        # them only once then make them available to all ants
        # to spare computation, at the cost of possibly not always finding valid tours, one can limit the size to
        # num_cities x nn  as it is unlikely that far cities are ever preferred by ants
        # in the same fashion, eta^beta can be stored somewhere as these values don't change
        self.ants = None


class SingleAnt:
    def __init__(self, num_cities):
        self.n = num_cities
        self.tour_length = 0.0  # ant's tour length
        self.tour = []  # store the (partial) tour
        self.visited = [False for _ in range(self.n)]  # visited cities


class AS(ACO):
    """
    Ant System
    """

    def __init__(self, alpha = 1, beta = 2, rho = 0.5, nn = 20):
        super().__init__(alpha, beta, rho, nn)

    def __call__(self, TSP_instance: TSPProblem):
        # Data Initialization
        self.distance_matrix = TSP_instance.distance_matrix
        self.num_cities = TSP_instance.DIMENSION
        self.nn_list = TSP_instance.nn_list
        self.pheromone = self._initialize_pheromone_matrix(self.num_cities, TSP_instance.nn_length)
        self.choice_info = self._initialize_choice_info()
        print(self.num_cities)
        print(self.choice_info)

    def _initialize_pheromone_matrix(self, num_ants, nn_length):
        return np.ones(shape = (self.num_cities, self.num_cities)) * (num_ants / nn_length)

    def _initialize_choice_info(self):
        return np.power(self.pheromone, self.alpha) * np.power(self.distance_matrix, - self.beta)
