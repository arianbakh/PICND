import matplotlib.pyplot as plt
import networkx as nx
import os
import random
import warnings

from matplotlib.backends import backend_gtk3

from settings import OUTPUT_DIR


warnings.filterwarnings('ignore', module=backend_gtk3.__name__)


RESTART_PROBABILITY = 0.15
STEPS_MULTIPLIER = 100


def random_walk_sample(graph, n):
    selected_nodes = set()
    while len(selected_nodes) < n:
        last_node = random.choice(list(graph.nodes))
        selected_nodes.add(last_node)
        for i in range(STEPS_MULTIPLIER * n):
            last_node = random.choice(list(graph.neighbors(last_node)))
            selected_nodes.add(last_node)
            if len(selected_nodes) >= n:
                break
    subgraph = graph.subgraph(selected_nodes)
    return subgraph


def save_graph_figure(graph, name):
    plt.title(name, fontsize=16)
    nx.draw(graph, node_size=100)
    plt.savefig(os.path.join(OUTPUT_DIR, '%s.png' % name))
    plt.close('all')
