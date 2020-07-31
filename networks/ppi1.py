import gzip
import networkx as nx
import os
import shutil
import sys
import urllib.request

from networks.abstract_network import Network
from networks.tools import random_walk_sample, save_graph_figure
from settings import DATA_DIR


PPI1_URL = 'https://stringdb-static.org/download/protein.links.v11.0/4932.protein.links.v11.0.txt.gz'
PPI1_TAR_PATH = os.path.join(DATA_DIR, '4932.protein.links.v11.0.txt.gz')
PPI1_TXT_PATH = os.path.join(DATA_DIR, '4932.protein.links.v11.0.txt')


class PPI1(Network):
    name = 'PPI1'

    def __init__(self):
        self.sample_size = 20
        super().__init__()

    @staticmethod
    def _ensure_data():
        if not os.path.exists(PPI1_TXT_PATH):
            if not os.path.exists(PPI1_TAR_PATH):
                opener = urllib.request.build_opener()
                opener.addheaders = [('authority', 'stringdb-static.org')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(PPI1_URL, PPI1_TAR_PATH)
            with gzip.open(PPI1_TAR_PATH, 'rb') as f_in:
                with open(PPI1_TXT_PATH, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    @staticmethod
    def _data_generator():
        PPI1._ensure_data()
        with open(PPI1_TXT_PATH, 'r') as txt_file:
            for i, line in enumerate(txt_file.readlines()):
                # progress bar
                if i % 1000 == 0:
                    sys.stdout.write('\r[%d]' % i)
                    sys.stdout.flush()

                if i > 0:
                    split_line = line.strip().split()
                    from_id = split_line[0]
                    to_id = split_line[1]
                    weight = int(split_line[2])
                    yield from_id, to_id, weight
            print()  # newline

    def _create_adjacency_matrix(self):
        graph = nx.Graph()
        for from_id, to_id, weight in PPI1._data_generator():
            graph.add_edge(from_id, to_id)

        subgraph = random_walk_sample(graph, self.sample_size)

        self.number_of_nodes = subgraph.number_of_nodes()
        save_graph_figure(subgraph, PPI1.name)
        return nx.to_numpy_array(subgraph)
