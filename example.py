import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings

from matplotlib import rc
from matplotlib.backends import backend_gtk3

from settings import OUTPUT_DIR


warnings.filterwarnings('ignore', module=backend_gtk3.__name__)


ADJACENCY_MATRIX = np.array([[0, 0, 1], [1, 0, 0], [1, 1, 0]])
INITIAL_VALUES = np.array([0.0, 0.0, 1.0])
TOTAL_TIME = 5.0
NUMBER_OF_NODES = ADJACENCY_MATRIX.shape[0]


def _calculate_self_dynamics(last_state):
    result = []
    for node_index in range(NUMBER_OF_NODES):
        result.append(-1 * last_state[node_index])
    return np.array(result)


def _calculate_neighbor_dynamics(last_state):
    result = []
    for node_index in range(NUMBER_OF_NODES):
        sum_result = 0
        for neighbor_index in range(NUMBER_OF_NODES):
            if ADJACENCY_MATRIX[node_index, neighbor_index]:
                sum_result += (1 - last_state[node_index]) * last_state[neighbor_index]
        result.append(sum_result)
    return np.array(result)


def _draw_node_plot(self_dynamics, neighbor_dynamics, node_index, frames):
    data_frame = pd.DataFrame({
        'iterations': np.arange(self_dynamics.shape[0]),
        'Self-Dynamics': self_dynamics,
        'Neighbor Dynamics': neighbor_dynamics,
        'Derivative': self_dynamics + neighbor_dynamics
    })
    melted_data_frame = pd.melt(
        data_frame,
        id_vars=['iterations'],
        value_vars=[
            'Self-Dynamics',
            'Neighbor Dynamics',
            'Derivative'
        ]
    )
    rc('font', weight=500)
    plt.subplots(figsize=(11, 6))
    ax = sns.lineplot(x='iterations', y='value', hue='variable', data=melted_data_frame, linewidth=4, palette=["C0", "C1", "k"])
    ax.set_title('Derivative of Node %d' % (node_index + 1), fontsize=28, fontweight=500)
    ax.set_xlabel('Time Frames', fontsize=20, fontweight=500)
    ax.set_ylabel('', fontsize=20, fontweight=500)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3, length=10, labelsize=16)
    plt.legend(prop={'size': 20, 'weight': 'normal'}, loc='upper left', bbox_to_anchor=(0.5, -0.15))
    plt.savefig(os.path.join(OUTPUT_DIR, 'node_%d_frames_%d.png' % (node_index, frames)), bbox_inches='tight')
    plt.close('all')


def _draw_plot(frames):
    delta_t = TOTAL_TIME / frames
    states = [INITIAL_VALUES]
    self_dynamics = []
    neighbor_dynamics = []
    for frame in range(frames):
        self_dynamics.append(_calculate_self_dynamics(states[-1]))
        neighbor_dynamics.append(_calculate_neighbor_dynamics(states[-1]))
        states.append(states[-1] + self_dynamics[-1] * delta_t + neighbor_dynamics[-1] * delta_t)
    self_dynamics = np.array(self_dynamics)
    neighbor_dynamics = np.array(neighbor_dynamics)
    for node_index in range(NUMBER_OF_NODES):
        _draw_node_plot(self_dynamics[:, node_index], neighbor_dynamics[:, node_index], node_index, frames)


def draw_plots():
    for frames in [10, 100, 1000]:
        _draw_plot(frames)


if __name__ == '__main__':
    draw_plots()
