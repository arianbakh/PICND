from subprocess import call

from dynamic_models.epidemic_dynamic_model import EpidemicDynamicModel
from dynamic_models.population_dynamic_model import PopulationDynamicModel
from dynamic_models.regulatory_dynamic_model_1 import RegulatoryDynamicModel1
from dynamic_models.regulatory_dynamic_model_2 import RegulatoryDynamicModel2
from networks.uci_online import UCIOnline
from networks.eco1 import ECO1
from networks.eco2 import ECO2
from networks.ppi1 import PPI1

if __name__ == '__main__':
    call([
        'python3',
        'algorithm.py',
        UCIOnline.name,
        EpidemicDynamicModel.name,
    ])
    call([
        'python3',
        'algorithm.py',
        PPI1.name,
        RegulatoryDynamicModel1.name,
    ])
    call([
        'python3',
        'algorithm.py',
        PPI1.name,
        RegulatoryDynamicModel2.name,
    ])
    call([
        'python3',
        'algorithm.py',
        ECO2.name,
        PopulationDynamicModel.name,
    ])
    call([
        'python3',
        'algorithm.py',
        ECO1.name,
        PopulationDynamicModel.name,
    ])
