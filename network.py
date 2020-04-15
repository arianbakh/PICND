import numpy as np
import os

from sklearn.metrics.cluster import normalized_mutual_info_score

from settings import DEBUG_DIR


class Network:
    def __init__(self, x, time_frame_labels, node_labels):
        self.x = x
        self._normalize()
        self.time_frame_labels = time_frame_labels
        self.node_labels = node_labels
        self.adjacency_matrix = np.zeros((self.x.shape[1], self.x.shape[1]))
        self._calculate_adjacency_matrix()

    def _normalize(self):
        normalized_columns = []
        for column_index in range(self.x.shape[1]):
            column = self.x[:, column_index]
            std = max(10 ** -9, np.std(column))  # to avoid division by zero
            normalized_column = (column - np.mean(column)) / std
            normalized_columns.append(normalized_column)
        normalized_x = np.column_stack(normalized_columns)
        self.x = normalized_x

    def _calculate_adjacency_matrix(self):
        for i in range(self.x.shape[1]):
            x_i = self.x[:, i]
            for j in range(i + 1, self.x.shape[1]):
                x_j = self.x[:, j]
                nmi = normalized_mutual_info_score(x_i, x_j)
                self.adjacency_matrix[i, j] = nmi
                self.adjacency_matrix[j, i] = nmi

    def _debug(self):
        np.savetxt(
            os.path.join(DEBUG_DIR, 'adjacency_matrix.csv'),
            self.adjacency_matrix,
            delimiter=',',
            fmt='%.2f'
        )
