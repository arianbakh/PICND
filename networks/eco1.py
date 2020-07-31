import networkx as nx
import numpy as np

from networks.abstract_network import Network
from settings import PLANT_POLLINATOR_CSV_PATH


class ECO1(Network):
    name = 'ECO1'

    def __init__(self):
        super().__init__()

    def _create_adjacency_matrix(self):
        plant_pollinator_list = []
        with open(PLANT_POLLINATOR_CSV_PATH, 'r') as csv_file:
            for line in csv_file.readlines():
                split_line = line.strip().split(',')
                plant_pollinator_list.append([int(item) for item in split_line])
        m = np.array(plant_pollinator_list)
        plants = m.shape[0]
        pollinators = m.shape[1]
        b = np.zeros((pollinators, pollinators))
        for k in range(pollinators):
            for l in range(k + 1, pollinators):
                total_weight = 0
                for i in range(plants):
                    total_weight += m[i, k] * m[i, l] / np.sum(m[i])
                b[k, l] = total_weight
                b[l, k] = total_weight
        graph = nx.from_numpy_matrix(b)
        sorted_components = sorted(nx.connected_components(graph), key=len, reverse=True)
        giant_component = graph.subgraph(sorted_components[0])
        self.number_of_nodes = len(giant_component)
        return nx.to_numpy_array(giant_component)
