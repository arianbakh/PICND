import numpy as np
import random

from networks.abstract_network import Network


class FullyConnectedRandomWeights(Network):
    def _create_adjacency_matrix(self):
        a = np.zeros((self.number_of_nodes, self.number_of_nodes))
        for i in range(self.number_of_nodes):
            for j in range(self.number_of_nodes):
                if i != j:
                    a[i, j] = random.random()
        return a
