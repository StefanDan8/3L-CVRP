import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class TSPProblem:
    def __init__(self, name, comment, dimension, weight_type, graph):
        self.NAME = name
        self.COMMENT = comment
        self.DIMENSION = dimension
        self.EDGE_WEIGHT_TYPE = weight_type
        self.graph = graph
        self.distance_matrix = self._compute_distances()
        self.nn_list = self._compute_nearest_neighbor_list()
        self.nn_length = self._nearest_neighbor_tour()

    def visualize(self):
        nx.draw(self.graph, nx.get_node_attributes(self.graph, 'pos'), with_labels = True,
                node_size = 0, font_size = 6, edgelist = [])
        plt.show()

    def _compute_distances(self):
        distance_matrix = np.zeros((self.DIMENSION, self.DIMENSION))
        for i in range(self.DIMENSION):
            for j in range(i + 1, self.DIMENSION):
                dist = _euclid(self.graph.nodes[i]['pos'], self.graph.nodes[j]['pos'])
                distance_matrix[i, j] = dist
                self.graph.add_edge(i, j, weight = dist)
            distance_matrix[i, i] = np.Inf
        distance_matrix = distance_matrix + np.tril(distance_matrix.T, -1)
        return distance_matrix

    def _compute_nearest_neighbor_list(self):
        return [np.argsort(self.distance_matrix[i, :]) for i in range(self.DIMENSION)]

    def _nearest_neighbor_tour(self):
        nn_tour = nx.algorithms.approximation.greedy_tsp(self.graph)
        length = 0.0
        for i in range(self.DIMENSION):
            length += self.distance_matrix[nn_tour[i], nn_tour[i + 1]]
        return length


def _euclid(x, y):
    return np.round(np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2), 2)


class TSPReader:
    def __init__(self):
        pass

    def __call__(self, path):
        with open(path, 'r') as f:
            name = f.readline().strip().split(': ')[1]
            f.readline()  # no need for type. All are tsp so far
            comment = f.readline().strip().split(': ')[1]
            dimension = int(f.readline().strip().split()[1])
            weight_type = f.readline().strip().split(': ')[1]
            f.readline()
            graph = nx.Graph()
            for i in range(dimension):
                line = f.readline().strip().split()
                graph.add_node(int(line[0]) - 1, pos = (float(line[1]), float(line[2])))
            return TSPProblem(name, comment, dimension, weight_type, graph)


def main():
    reader = TSPReader()
    berlin = reader('../tsplib-master/berlin52.tsp')
    berlin.visualize()
    # america = reader('../tsplib-master/att532.tsp')
    # america.visualize()


if __name__ == '__main__':
    main()
