from abc import ABC, abstractmethod


class DynamicModel(ABC):
    def __init__(self, network, delta_t):
        self.network = network
        self.delta_t = delta_t

    @abstractmethod
    def get_x(self, time_frames):
        pass

    def get_x_dot(self, x):
        x_dot = (x[1:] - x[:len(x) - 1]) / self.delta_t
        return x_dot
