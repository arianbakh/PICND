from abc import ABC, abstractmethod


class Network(ABC):
    def __init__(self):
        self.adjacency_matrix = self._create_adjacency_matrix()

    @abstractmethod
    def _create_adjacency_matrix(self):
        pass
