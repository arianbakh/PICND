import os


# Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


# Algorithm Settings
TIME_FRAMES = 100
D3CND_CHROMOSOME_SIZE = 5
GA_CHROMOSOME_SIZE = 10
GENE_SIZE = 12  # bits
MUTATION_CHANCE = 0.1
POPULATION = 100
CHILDREN = 10
TERMINATION_CONDITION = 500  # number of iterations allowed without improvement
POWER_RANGE = (0, 2)
COEFFICIENT_RANGE_OFFSET = 1
STEP = (POWER_RANGE[1] - POWER_RANGE[0]) / 2 ** GENE_SIZE
GA_METHOD_NAME = 'GA'
D3CND_METHOD_NAME = 'D3CND'
PLANT_POLLINATOR_CSV_PATH = os.path.join(DATA_DIR, 'M_PL_007.csv')
