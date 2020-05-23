import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import sys
import time
import warnings

from matplotlib.backends import backend_gtk3

from dynamic_models.epidemic_dynamic_model import EpidemicDynamicModel
from dynamic_models.synthetic_dynamic_model_1 import SyntheticDynamicModel1
from networks.fully_connected_random_weights import FullyConnectedRandomWeights
from networks.uci_online import UCIOnline
from settings import OUTPUT_DIR, TIME_FRAMES, CHROMOSOME_SIZE, GENE_SIZE, MUTATION_CHANCE, POPULATION, CHILDREN, \
    TERMINATION_CONDITION, POWER_RANGE, STEP

warnings.filterwarnings('ignore', module=backend_gtk3.__name__)
sns.set()


def _get_theta(x, adjacency_matrix, powers):
    theta_list = []
    for node_index in range(x.shape[1]):
        x_i = x[:TIME_FRAMES, node_index]
        column_list = [
            np.ones(TIME_FRAMES),
            x_i ** powers[0],
        ]

        ij_terms = []
        j_terms = []
        for j in range(x.shape[1]):
            if j != node_index and adjacency_matrix[j, node_index]:
                x_j = x[:TIME_FRAMES, j]
                ij_terms.append(adjacency_matrix[j, node_index] * x_i ** powers[1] * x_j ** powers[2])
                j_terms.append(adjacency_matrix[j, node_index] * x_j ** powers[3])
        if ij_terms:
            ij_column = np.sum(ij_terms, axis=0)
            column_list.append(ij_column)
        else:
            column_list.append(np.zeros(TIME_FRAMES))
        if j_terms:
            j_column = np.sum(j_terms, axis=0)
            column_list.append(j_column)
        else:
            column_list.append(np.zeros(TIME_FRAMES))

        theta = np.column_stack(column_list)
        theta_list.append(theta)
    return np.concatenate(theta_list)


def _get_complete_individual(x, y, adjacency_matrix, chromosome):
    powers = []
    for i in range(CHROMOSOME_SIZE):
        binary = 0
        for j in range(GENE_SIZE):
            binary += chromosome[i * GENE_SIZE + j] * 2 ** (GENE_SIZE - j - 1)
        power = POWER_RANGE[0] + binary * STEP
        powers.append(power)
    theta = _get_theta(x, adjacency_matrix, powers)
    stacked_y = np.concatenate([y[:, node_index] for node_index in range(y.shape[1])])
    coefficients = np.linalg.lstsq(theta, stacked_y, rcond=None)[0]
    y_hat = np.matmul(theta, coefficients.T)
    mse = np.mean((stacked_y - y_hat) ** 2)
    return {
        'chromosome': chromosome,
        'powers': powers,
        'coefficients': coefficients,
        'mse': mse,
        'fitness': 1 / mse,
    }


class Population:
    def __init__(self, size, x, y, adjacency_matrix):
        self.size = size
        self.x = x
        self.y = y
        self.adjacency_matrix = adjacency_matrix
        self.individuals = self._get_complete_individuals(self._get_initial_individuals())

    def _get_complete_individuals(self, individuals):
        complete_individuals = []
        for individual in individuals:
            complete_individuals.append(_get_complete_individual(
                self.x,
                self.y,
                self.adjacency_matrix,
                individual['chromosome'],
            ))
        return complete_individuals

    def _get_initial_individuals(self):
        individuals = []
        for i in range(self.size):
            individuals.append(
                {
                    'chromosome': [random.randint(0, 1) for _ in range(CHROMOSOME_SIZE * GENE_SIZE)],
                }
            )
        return individuals

    @staticmethod
    def _crossover(chromosome1, chromosome2):
        crossover_point = random.randint(0, CHROMOSOME_SIZE * GENE_SIZE - 1)
        offspring_chromosome = chromosome1[:crossover_point] + chromosome2[crossover_point:]
        return offspring_chromosome

    @staticmethod
    def _mutation(chromosome):
        mutated_chromosome = []
        for i in range(CHROMOSOME_SIZE * GENE_SIZE):
            if random.random() < MUTATION_CHANCE:
                mutated_chromosome.append(0 if chromosome[i] else 1)
            else:
                mutated_chromosome.append(chromosome[i])
        return mutated_chromosome

    @staticmethod
    def _select_random_individual(sorted_individuals, total_fitness):
        random_value = random.random()
        selected_index = 0
        sum_fitness = sorted_individuals[0]['fitness']
        for i in range(1, len(sorted_individuals)):
            if sum_fitness / total_fitness > random_value:
                break
            selected_index = i
            sum_fitness += sorted_individuals[i]['fitness']
        return selected_index

    def run_single_iteration(self):
        # the following two values are pre-calculated to increase performance
        sorted_individuals = sorted(self.individuals, key=lambda individual: -1 * individual['fitness'])
        total_fitness = sum([individual['fitness'] for individual in self.individuals])

        children = []
        while len(children) < CHILDREN:
            individual1_index = Population._select_random_individual(sorted_individuals, total_fitness)
            individual2_index = Population._select_random_individual(sorted_individuals, total_fitness)
            if individual1_index != individual2_index:
                chromosome1 = self.individuals[individual1_index]['chromosome']
                chromosome2 = self.individuals[individual2_index]['chromosome']
                offspring_chromosome = Population._mutation(Population._crossover(chromosome1, chromosome2))
                children.append({
                    'chromosome': offspring_chromosome,
                })
        children = self._get_complete_individuals(children)

        new_individuals = sorted(self.individuals + children, key=lambda individual: -1 * individual['fitness'])
        self.individuals = new_individuals[:self.size]

        return self.individuals[0]  # fittest


def _draw_error_plot(errors, network_name, dynamic_model_name):
    data_frame = pd.DataFrame({
        'iterations': np.arange(len(errors)),
        'errors': np.array(errors),
    })
    plt.subplots(figsize=(20, 10))
    ax = sns.lineplot(x='iterations', y='errors', data=data_frame)
    ax.set(
        xlabel='Iteration',
        ylabel='log10(MSE) of Fittest Individual',
        title='%s model on %s network' % (dynamic_model_name, network_name)
    )
    plt.savefig(os.path.join(OUTPUT_DIR, '%s_on_%s.png' % (dynamic_model_name, network_name)))
    plt.close('all')


def run(network_name, dynamic_model_name):
    network = None
    if network_name == FullyConnectedRandomWeights.name:
        network = FullyConnectedRandomWeights()
    elif network_name == UCIOnline.name:
        network = UCIOnline()
    else:
        print('Invalid network name')
        exit(0)

    dynamic_model = None
    if dynamic_model_name == SyntheticDynamicModel1.name:
        dynamic_model = SyntheticDynamicModel1(network)
    elif dynamic_model_name == EpidemicDynamicModel.name:
        dynamic_model = EpidemicDynamicModel(network)
    else:
        print('Invalid dynamic model name')
        exit(0)

    x = dynamic_model.get_x(TIME_FRAMES)
    y = dynamic_model.get_x_dot(x)

    population = Population(POPULATION, x, y, network.adjacency_matrix)
    fittest_individual = None
    counter = 0
    best_fitness = 0
    best_index = 0
    errors = []
    start_time = time.time()
    while counter - best_index < TERMINATION_CONDITION:
        counter += 1
        fittest_individual = population.run_single_iteration()
        errors.append(math.log10(fittest_individual['mse']))
        if fittest_individual['fitness'] > best_fitness:
            best_fitness = fittest_individual['fitness']
            best_index = counter
        if counter % 100 == 0:
            print(fittest_individual['mse'])
    end_time = time.time()
    print('took', counter, 'iterations;', int(end_time - start_time), 'seconds')
    print('%f + %f * xi^%f + %f * sum Aij * xi^%f * xj^%f + %f * sum Aij * xj^%f' % (
        fittest_individual['coefficients'][0],
        fittest_individual['coefficients'][1],
        fittest_individual['powers'][0],
        fittest_individual['coefficients'][2],
        fittest_individual['powers'][1],
        fittest_individual['powers'][2],
        fittest_individual['coefficients'][3],
        fittest_individual['powers'][3]
    ))
    _draw_error_plot(errors, network_name, dynamic_model_name)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Invalid number of arguments')
        exit(0)
    run(sys.argv[1], sys.argv[2])
