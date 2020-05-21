import networkx as nx
import os
import tarfile
import urllib.request

from datetime import datetime, timedelta
from networks.abstract_network import Network
from settings import DATA_DIR


UCI_ONLINE_URL = 'http://konect.uni-koblenz.de/downloads/tsv/opsahl-ucsocial.tar.bz2'
UCI_ONLINE_TAR_PATH = os.path.join(DATA_DIR, 'opsahl-ucsocial.tar.bz2')
UCI_ONLINE_DIR = os.path.join(DATA_DIR, 'opsahl-ucsocial')
UCI_ONLINE_TSV_PATH = os.path.join(UCI_ONLINE_DIR, 'out.opsahl-ucsocial')


class UCIOnline(Network):
    name = 'UCI'

    def __init__(self):
        self.offset_days = 0
        self.included_days = 5
        super().__init__()

    @staticmethod
    def _ensure_data():
        if not os.path.exists(UCI_ONLINE_DIR):
            urllib.request.urlretrieve(UCI_ONLINE_URL, UCI_ONLINE_TAR_PATH)
            tar = tarfile.open(UCI_ONLINE_TAR_PATH, "r:bz2")
            tar.extractall(DATA_DIR)
            tar.close()

    @staticmethod
    def _data_generator():
        UCIOnline._ensure_data()
        with open(UCI_ONLINE_TSV_PATH, 'r') as tsv_file:
            for i, line in enumerate(tsv_file.readlines()):
                if not line.startswith('%'):
                    split_line = line.strip().split()
                    from_id = int(split_line[0])
                    to_id = int(split_line[1])
                    count = int(split_line[2])
                    timestamp = int(split_line[3])
                    yield from_id, to_id, count, timestamp

    def _create_adjacency_matrix(self):
        graph = nx.Graph()
        first_datetime = None
        for from_id, to_id, count, timestamp in UCIOnline._data_generator():
            if first_datetime is None:
                first_datetime = datetime.utcfromtimestamp(timestamp)
            else:
                current_datetime = datetime.utcfromtimestamp(timestamp)
                if first_datetime + timedelta(days=self.offset_days) < current_datetime < \
                        first_datetime + timedelta(days=self.offset_days + self.included_days):
                    graph.add_edge(from_id, to_id)
                elif current_datetime > first_datetime + timedelta(days=self.offset_days + self.included_days):
                    break
        self.number_of_nodes = graph.number_of_nodes()
        return nx.to_numpy_array(graph)
