import random


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
