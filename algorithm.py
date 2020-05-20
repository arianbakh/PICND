import numpy as np
import random

from dynamic_models.synthetic_dynamics_model_1 import SyntheticDynamicsModel1
from networks.fully_connected_random_weights import FullyConnectedRandomWeights


NUMBER_OF_NODES = 10
DELTA_T = 0.01
TIME_FRAMES = 10
CHROMOSOME_SIZE = 3
GENE_SIZE = 8  # bits
MUTATION_CHANCE = 0.1
POPULATION = 100
CHILDREN = 10
ITERATIONS = 3000
POWER_RANGE = (0, 2)


STEP = (POWER_RANGE[1] - POWER_RANGE[0]) / 2 ** GENE_SIZE


class Individual:
    def __init__(self, chromosome, x, y, adjacency_matrix):
        self.coefficients = None
        self.chromosome = chromosome
        self.fitness = self._calculate_fitness(x, y, adjacency_matrix)

    def _get_theta(self, x, adjacency_matrix):
        theta_list = []
        for node_index in range(NUMBER_OF_NODES):
            x_i = x[:TIME_FRAMES, node_index]
            column_list = [
                np.ones(TIME_FRAMES),
                x_i ** self.powers[0],
            ]
            terms = []
            for j in range(NUMBER_OF_NODES):
                if j != node_index and adjacency_matrix[j, node_index]:
                    x_j = x[:TIME_FRAMES, j]
                    terms.append(
                        adjacency_matrix[j, node_index] * x_i ** self.powers[1] * x_j ** self.powers[2])
            if terms:
                column = np.sum(terms, axis=0)
                column_list.append(column)
            theta = np.column_stack(column_list)
            theta_list.append(theta)
        return np.concatenate(theta_list)

    def _calculate_mse(self, x, y, adjacency_matrix):
        powers = []
        for i in range(CHROMOSOME_SIZE):
            binary = 0
            for j in range(GENE_SIZE):
                binary += self.chromosome[i * GENE_SIZE + j] * 2 ** (GENE_SIZE - j - 1)
            power = POWER_RANGE[0] + binary * STEP
            powers.append(power)
        self.powers = powers
        theta = self._get_theta(x, adjacency_matrix)
        stacked_y = np.concatenate([y[:, node_index] for node_index in range(NUMBER_OF_NODES)])
        coefficients = np.linalg.lstsq(theta, stacked_y, rcond=None)[0]
        self.coefficients = coefficients
        y_hat = np.matmul(theta, coefficients.T)
        return np.mean((stacked_y - y_hat) ** 2)

    def _calculate_least_difference(self):
        sorted_powers = np.sort(self.powers)
        return np.min(sorted_powers[1:] - sorted_powers[:-1])

    def _calculate_fitness(self, x, y, adjacency_matrix):
        mse = self._calculate_mse(x, y, adjacency_matrix)
        least_difference = self._calculate_least_difference()
        return least_difference / mse


class Population:
    def __init__(self, size, x, y, adjacency_matrix):
        self.size = size
        self.x = x
        self.y = y
        self.adjacency_matrix = adjacency_matrix
        self.individuals = self._initialize_individuals()

    def _initialize_individuals(self):
        individuals = []
        for i in range(self.size):
            individuals.append(Individual(
                [random.randint(0, 1) for _ in range(CHROMOSOME_SIZE * GENE_SIZE)],
                self.x,
                self.y,
                self.adjacency_matrix
            ))
        return individuals

    def _crossover(self, individual1, individual2):
        crossover_point = random.randint(0, CHROMOSOME_SIZE * GENE_SIZE - 1)
        offspring_chromosome = individual1.chromosome[:crossover_point] + individual2.chromosome[crossover_point:]
        return Individual(offspring_chromosome, self.x, self.y, self.adjacency_matrix)

    def _mutation(self, individual):
        mutated_chromosome = []
        for i in range(CHROMOSOME_SIZE * GENE_SIZE):
            if random.random() < MUTATION_CHANCE:
                mutated_chromosome.append(0 if individual.chromosome[i] else 1)
            else:
                mutated_chromosome.append(individual.chromosome[i])
        return Individual(mutated_chromosome, self.x, self.y, self.adjacency_matrix)

    @staticmethod
    def _select_random_individual(sorted_individuals, total_fitness):
        random_value = random.random()
        selected_index = 0
        selected_individual = sorted_individuals[0]
        sum_fitness = selected_individual.fitness
        for i in range(1, len(sorted_individuals)):
            if sum_fitness / total_fitness > random_value:
                break
            selected_index = i
            selected_individual = sorted_individuals[i]
            sum_fitness += selected_individual.fitness
        return selected_index, selected_individual

    def run_single_iteration(self):
        # the following two values are pre-calculated to increase performance
        sorted_individuals = sorted(self.individuals, key=lambda individual: -1 * individual.fitness)
        total_fitness = sum([individual.fitness for individual in self.individuals])

        children = []
        while len(children) < CHILDREN:
            individual1_index, individual1 = self._select_random_individual(sorted_individuals, total_fitness)
            individual2_index, individual2 = self._select_random_individual(sorted_individuals, total_fitness)
            if individual1_index != individual2_index:
                children.append(self._mutation(self._crossover(individual1, individual2)))

        new_individuals = sorted(self.individuals + children, key=lambda individual: -1 * individual.fitness)
        self.individuals = new_individuals[:self.size]

        return self.individuals[0]  # fittest


def run():
    network = FullyConnectedRandomWeights(NUMBER_OF_NODES)
    adjacency_matrix = network.adjacency_matrix

    dynamics_model = SyntheticDynamicsModel1(network, DELTA_T)
    x = dynamics_model.get_x(TIME_FRAMES)
    y = dynamics_model.get_x_dot(x)

    population = Population(POPULATION, x, y, adjacency_matrix)  # TODO don't pass adjacency matrix
    fittest_individual = None
    for i in range(ITERATIONS):
        fittest_individual = population.run_single_iteration()
        if i % 1000 == 0:
            print(1 / fittest_individual.fitness)
    print('%f + %f * xi^%f + %f * sum Aij * xi^%f * xj^%f' % (
        fittest_individual.coefficients[0],
        fittest_individual.coefficients[1],
        fittest_individual.powers[0],
        fittest_individual.coefficients[2],
        fittest_individual.powers[1],
        fittest_individual.powers[2]
    ))


if __name__ == '__main__':
    run()
