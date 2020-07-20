import numpy as np

from dynamic_models.abstract_dynamic_model import DynamicModel


class PopulationDynamicModel(DynamicModel):
    name = 'P'

    def __init__(self, network):
        super().__init__(network, delta_t=0.01)
        self.offset_time_frames = 100

    def get_x(self, time_frames):
        number_of_nodes = self.network.number_of_nodes
        adjacency_matrix = self.network.adjacency_matrix
        total_time_frames = self.offset_time_frames + time_frames + 1

        x = np.zeros((total_time_frames, number_of_nodes))
        x[0] = np.ones(number_of_nodes)
        for i in range(1, total_time_frames):
            for j in range(number_of_nodes):
                f_result = -1 * (x[i - 1, j] ** 0.5)
                g_result = 0
                for k in range(number_of_nodes):
                    if k != j:
                        g_result += adjacency_matrix[k, j] * (x[i - 1, k] ** 0.2)
                derivative = f_result + g_result
                x[i, j] = x[i - 1, j] + self.delta_t * derivative
        return x[-(time_frames + 1):]
