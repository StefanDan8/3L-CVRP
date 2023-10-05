from ACO import AS
from tsp_reader import TSPReader, TSPProblem
aco = AS()
reader = TSPReader()
berlin = reader('../tsplib-master/berlin52.tsp')
aco(berlin)
