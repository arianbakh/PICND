from subprocess import call

from dynamic_models.epidemic_dynamic_model import EpidemicDynamicModel
from dynamic_models.population_dynamic_model import PopulationDynamicModel
from dynamic_models.regulatory_dynamic_model_1 import RegulatoryDynamicModel1
from dynamic_models.regulatory_dynamic_model_2 import RegulatoryDynamicModel2
from dynamic_models.synthetic_dynamic_model_1 import SyntheticDynamicModel1
from networks.fully_connected_random_weights import FullyConnectedRandomWeights
from networks.uci_online import UCIOnline
from settings import D3CND_METHOD_NAME, GA_METHOD_NAME

if __name__ == '__main__':
    for network in [
        FullyConnectedRandomWeights,
        UCIOnline,
    ]:
        for dynamic_model in [
            EpidemicDynamicModel,
            PopulationDynamicModel,
            RegulatoryDynamicModel1,
            RegulatoryDynamicModel2,
            SyntheticDynamicModel1,
        ]:
            for method_name in [
                D3CND_METHOD_NAME,
                GA_METHOD_NAME,
            ]:
                call([
                    'python3',
                    'algorithm.py',
                    network.name,
                    dynamic_model.name,
                    method_name,
                ])
