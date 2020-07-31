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

from matplotlib import rc
from matplotlib.backends import backend_gtk3

from dynamic_models.epidemic_dynamic_model import EpidemicDynamicModel
from dynamic_models.population_dynamic_model import PopulationDynamicModel
from dynamic_models.regulatory_dynamic_model_1 import RegulatoryDynamicModel1
from dynamic_models.regulatory_dynamic_model_2 import RegulatoryDynamicModel2
from dynamic_models.synthetic_dynamic_model_1 import SyntheticDynamicModel1
from networks.fully_connected_random_weights import FullyConnectedRandomWeights
from networks.uci_online import UCIOnline
from networks.eco1 import ECO1
from networks.eco2 import ECO2
from settings import OUTPUT_DIR, TIME_FRAMES, D3CND_CHROMOSOME_SIZE, GA_CHROMOSOME_SIZE, GENE_SIZE, MUTATION_CHANCE, \
    POPULATION, CHILDREN, TERMINATION_CONDITION, POWER_RANGE, COEFFICIENT_RANGE_OFFSET, STEP, GA_METHOD_NAME, \
    D3CND_METHOD_NAME

warnings.filterwarnings('ignore', module=backend_gtk3.__name__)


def _get_theta(x, adjacency_matrix, powers):
    theta_list = []
    for node_index in range(x.shape[1]):
        x_i = x[:TIME_FRAMES, node_index]
        column_list = [
            np.ones(TIME_FRAMES),
            x_i ** powers[0],
        ]

        first_ij_terms = []
        first_j_terms = []
        second_j_terms = []
        for j in range(x.shape[1]):
            if j != node_index and adjacency_matrix[j, node_index]:
                x_j = x[:TIME_FRAMES, j]
                first_ij_terms.append(adjacency_matrix[j, node_index] * x_i ** powers[1] * x_j ** powers[2])
                first_j_terms.append(adjacency_matrix[j, node_index] * x_j ** powers[3])
                second_j_terms.append(adjacency_matrix[j, node_index] * (1 - 1 / (1 + x_j ** powers[4])))
        if first_ij_terms:
            first_ij_column = np.sum(first_ij_terms, axis=0)
            column_list.append(first_ij_column)
        else:
            column_list.append(np.zeros(TIME_FRAMES))
        if first_j_terms:
            first_j_column = np.sum(first_j_terms, axis=0)
            column_list.append(first_j_column)
        else:
            column_list.append(np.zeros(TIME_FRAMES))
        if second_j_terms:
            second_j_column = np.sum(second_j_terms, axis=0)
            column_list.append(second_j_column)
        else:
            column_list.append(np.zeros(TIME_FRAMES))

        theta = np.column_stack(column_list)
        theta_list.append(theta)
    return np.concatenate(theta_list)


def _get_complete_individual_pure_ga(x, y, adjacency_matrix, chromosome):
    numbers = []
    for i in range(GA_CHROMOSOME_SIZE):
        binary = 0
        for j in range(GENE_SIZE):
            binary += chromosome[i * GENE_SIZE + j] * 2 ** (GENE_SIZE - j - 1)
        if i in [0, 1, 3, 6, 8]:  # coefficients
            number = POWER_RANGE[0] + binary * STEP - COEFFICIENT_RANGE_OFFSET
        else:  # powers
            number = POWER_RANGE[0] + binary * STEP
        numbers.append(number)

    y_i_hats = []
    for i in range(x.shape[1]):
        x_i = x[:, i][:TIME_FRAMES]
        y_i_hat = numbers[0] + numbers[1] * x_i ** numbers[2]
        for j in range(x.shape[1]):
            if i != j and adjacency_matrix[j, i]:
                x_j = x[:, j][:TIME_FRAMES]
                y_i_hat += numbers[3] * adjacency_matrix[j, i] * (x_i ** numbers[4]) * (x_j ** numbers[5])
                y_i_hat += numbers[6] * adjacency_matrix[j, i] * (x_j ** numbers[7])
                y_i_hat += numbers[8] * adjacency_matrix[j, i] * (1 - 1 / (1 + x_j ** numbers[9]))
        y_i_hats.append(y_i_hat)
    y_hat = np.concatenate(y_i_hats)

    stacked_y = np.concatenate([y[:, node_index] for node_index in range(y.shape[1])])
    mse = np.mean((stacked_y - y_hat) ** 2)
    return {
        'chromosome': chromosome,
        'numbers': numbers,
        'mse': mse,
        'fitness': 1 / mse,
    }


def _get_complete_individual_d3cnd(x, y, adjacency_matrix, chromosome):
    powers = []
    for i in range(D3CND_CHROMOSOME_SIZE):
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
    numbers = [
        coefficients[0],
        coefficients[1],
        powers[0],
        coefficients[2],
        powers[1],
        powers[2],
        coefficients[3],
        powers[3],
        coefficients[4],
        powers[4],
    ]
    return {
        'chromosome': chromosome,
        'numbers': numbers,
        'mse': mse,
        'fitness': 1 / mse,
    }


class Population:
    def __init__(self, size, x, y, adjacency_matrix, method_name):
        self.size = size
        self.x = x
        self.y = y
        self.adjacency_matrix = adjacency_matrix

        self.method_name = method_name
        self.get_complete_individual = None
        self.chromosome_size = None
        if self.method_name == GA_METHOD_NAME:
            self.get_complete_individual = _get_complete_individual_pure_ga
            self.chromosome_size = GA_CHROMOSOME_SIZE
        elif self.method_name == D3CND_METHOD_NAME:
            self.get_complete_individual = _get_complete_individual_d3cnd
            self.chromosome_size = D3CND_CHROMOSOME_SIZE
        else:
            print('Invalid method name')
            exit(0)

        self.individuals = self._get_complete_individuals(self._get_initial_individuals())

    def _get_complete_individuals(self, individuals):
        complete_individuals = []
        for individual in individuals:
            complete_individuals.append(self.get_complete_individual(
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
                    'chromosome': [random.randint(0, 1) for _ in range(self.chromosome_size * GENE_SIZE)],
                }
            )
        return individuals

    def _crossover(self, chromosome1, chromosome2):
        crossover_point = random.randint(0, self.chromosome_size * GENE_SIZE - 1)
        offspring_chromosome = chromosome1[:crossover_point] + chromosome2[crossover_point:]
        return offspring_chromosome

    def _mutation(self, chromosome):
        mutated_chromosome = []
        for i in range(self.chromosome_size * GENE_SIZE):
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
                offspring_chromosome = self._mutation(self._crossover(chromosome1, chromosome2))
                children.append({
                    'chromosome': offspring_chromosome,
                })
        children = self._get_complete_individuals(children)

        new_individuals = sorted(self.individuals + children, key=lambda individual: -1 * individual['fitness'])
        self.individuals = new_individuals[:self.size]

        return self.individuals[0]  # fittest


def _draw_error_plot(errors, network_name, dynamic_model_name, method_name):
    data_frame = pd.DataFrame({
        'iterations': np.arange(len(errors)),
        'errors': np.array(errors),
    })
    rc('font', weight=600)
    plt.subplots(figsize=(11, 6))
    ax = sns.lineplot(x='iterations', y='errors', data=data_frame, linewidth=4)
    ax.set_title('%s on %s via %s' % (dynamic_model_name, network_name, method_name), fontsize=28, fontweight=600)
    ax.set_xlabel('Iteration', fontsize=20, fontweight=600)
    ax.set_ylabel('$log_{10}(MSE)$', fontsize=20, fontweight=600)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3, length=10, labelsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, '%s_on_%s_via_%s.png' % (dynamic_model_name, network_name, method_name)))
    plt.close('all')


def _save_info(exec_time, iterations, fittest_individual, network_name, dynamic_model_name, method_name):
    info_file_path = os.path.join(OUTPUT_DIR, '%s_on_%s_via_%s.txt' % (dynamic_model_name, network_name, method_name))
    with open(info_file_path, 'w') as info_file:
        info_file.write('exec_time: %ds\n' % exec_time)
        info_file.write('iterations: %d\n' % iterations)
        info_file.write('fittest_mse: %2e\n' % fittest_individual['mse'])
        info_file.write(
            '%f + %f * xi^%f + %f * sum Aij * xi^%f * xj^%f + %f * sum Aij * xj^%f + %f * sum Aij (1 - 1 / (1 + xj^%f))'
            % tuple(fittest_individual['numbers'])
        )
    with open(info_file_path, 'r') as info_file:
        print(info_file.read())


def run(network_name, dynamic_model_name, method_name):
    network = None
    if network_name == FullyConnectedRandomWeights.name:
        network = FullyConnectedRandomWeights()
    elif network_name == UCIOnline.name:
        network = UCIOnline()
    elif network_name == ECO1.name:
        network = ECO1()
    elif network_name == ECO2.name:
        network = ECO2()
    else:
        print('Invalid network name')
        exit(0)

    dynamic_model = None
    if dynamic_model_name == SyntheticDynamicModel1.name:
        dynamic_model = SyntheticDynamicModel1(network)
    elif dynamic_model_name == EpidemicDynamicModel.name:
        dynamic_model = EpidemicDynamicModel(network)
    elif dynamic_model_name == PopulationDynamicModel.name:
        dynamic_model = PopulationDynamicModel(network)
    elif dynamic_model_name == RegulatoryDynamicModel1.name:
        dynamic_model = RegulatoryDynamicModel1(network)
    elif dynamic_model_name == RegulatoryDynamicModel2.name:
        dynamic_model = RegulatoryDynamicModel2(network)
    else:
        print('Invalid dynamic model name')
        exit(0)

    x = dynamic_model.get_x(TIME_FRAMES)
    y = dynamic_model.get_x_dot(x)

    population = Population(POPULATION, x, y, network.adjacency_matrix, method_name)
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
    _draw_error_plot(errors, network_name, dynamic_model_name, method_name)
    _save_info(int(end_time - start_time), counter, fittest_individual, network_name, dynamic_model_name, method_name)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Invalid number of arguments')
        exit(0)
    run(sys.argv[1], sys.argv[2], sys.argv[3])
