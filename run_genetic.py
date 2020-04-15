import numpy as np
import random

from scipy.stats import iqr

from iran_stock import get_iran_stock_networks


# Genetic Settings
CHROMOSOME_SIZE = 4
GENE_SIZE = 12  # bits
MUTATION_CHANCE = 0.1
POPULATION = 100
CHILDREN = 10
ITERATIONS = 1000
POWER_RANGE = (-6, 10)


# Calculated Settings
STEP = (POWER_RANGE[1] - POWER_RANGE[0]) / 2 ** GENE_SIZE


class Individual:
    def __init__(self, chromosome, x, y, adjacency_matrix):
        self.coefficients = None
        self.chromosome = chromosome
        self.fitness = self._calculate_fitness(x, y, adjacency_matrix)

    def _get_theta(self, x, adjacency_matrix):
        number_of_nodes = x.shape[1]
        time_frames = x.shape[0] - 1

        theta_list = []
        for node_index in range(number_of_nodes):
            x_i = x[:time_frames, node_index]
            adjacency_sum = np.sum(adjacency_matrix[:, node_index])
            column_list = [
                np.ones(time_frames),
                x_i ** self.powers[0],
                adjacency_sum * x_i ** self.powers[1],
            ]
            terms = []
            for j in range(number_of_nodes):
                if j != node_index and adjacency_matrix[j, node_index]:
                    x_j = x[:time_frames, j]
                    terms.append(
                        adjacency_matrix[j, node_index] * x_i ** self.powers[2] * x_j ** self.powers[3])
            if terms:
                column = np.sum(terms, axis=0)
                column_list.append(column)
            theta = np.column_stack(column_list)
            theta_list.append(theta)
        return np.concatenate(theta_list)

    def _calculate_mse(self, x, y, adjacency_matrix):
        number_of_nodes = x.shape[1]

        powers = []
        for i in range(CHROMOSOME_SIZE):
            binary = 0
            for j in range(GENE_SIZE):
                binary += self.chromosome[i * GENE_SIZE + j] * 2 ** (GENE_SIZE - j - 1)
            power = POWER_RANGE[0] + binary * STEP
            powers.append(power)
        self.powers = powers
        theta = self._get_theta(x, adjacency_matrix)
        stacked_y = np.concatenate([y[:, node_index] for node_index in range(number_of_nodes)])
        coefficients = np.linalg.lstsq(theta, stacked_y, rcond=None)[0]
        self.coefficients = coefficients
        y_hat = np.matmul(theta, coefficients.T)
        return np.mean((stacked_y - y_hat) ** 2)

    def _calculate_fitness(self, x, y, adjacency_matrix):
        mse = self._calculate_mse(x, y, adjacency_matrix)
        return 1 / mse


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


def _get_y(x):
    x_dot = (x[1:] - x[:len(x) - 1])
    return x_dot


def run():
    iran_stock_networks = get_iran_stock_networks()
    sorted_iran_stock_networks = sorted(iran_stock_networks, key=lambda network: -network.dynamicity)
    x = sorted_iran_stock_networks[0].x
    x = x - np.min(x) + 1  # all entries are at least 1
    adjacency_matrix = sorted_iran_stock_networks[0].adjacency_matrix
    y = _get_y(x)
    interquartile_range = iqr(y)
    population = Population(POPULATION, x, y, adjacency_matrix)
    fittest_individual = None
    for i in range(ITERATIONS):
        fittest_individual = population.run_single_iteration()
        mse = 1 / fittest_individual.fitness
        rmsdiqr = mse ** 0.5 / interquartile_range
        print(rmsdiqr)
    print('%f + %f * xi^%f + %f * sum Aij xi^%f + %f * sum Aij * xi^%f * xj^%f' % (
        fittest_individual.coefficients[0],
        fittest_individual.coefficients[1],
        fittest_individual.powers[0],
        fittest_individual.coefficients[2],
        fittest_individual.powers[1],
        fittest_individual.coefficients[3],
        fittest_individual.powers[2],
        fittest_individual.powers[3]
    ))


if __name__ == '__main__':
    run()
