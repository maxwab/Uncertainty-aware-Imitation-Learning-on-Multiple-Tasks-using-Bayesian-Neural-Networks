#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np, os, sys, _pickle as pickle

import sys
#sys.path.insert(0,'./../Task_Agnostic_Online_Multitask_Imitation_Learning/')
sys.path.insert(0,'./../')
from Housekeeping import *


def plot_uncertainty_vs_tasks(all_task_configurations):
    fig = plt.figure()
    ax_1 = fig.add_subplot(121, frameon=False)
    ax_2 = fig.add_subplot(122, frameon=False)
    for iterator, configuration in enumerate(all_task_configurations):
        file_to_load_data_from = './' + configuration[EXPERIMENT_ID_KEY] + '/' + LOGS_DIRECTORY + configuration[CONTROLLER_KEY] + '_' + configuration[CONTEXTS_KEY] + '_' + configuration[WINDOW_SIZE_KEY] + '_' + configuration[PARTIAL_OBSERVABILITY_KEY] + '.pkl'
        configuration_label = configuration[CONTROLLER_KEY] + '_' + configuration[CONTEXTS_KEY] + '_' + configuration[WINDOW_SIZE_KEY] + '_' + configuration[PARTIAL_OBSERVABILITY_KEY]
        with open(file_to_load_data_from, 'rb') as f:
            loaded_data = pickle.load(f)
        all_tasks = list(loaded_data.keys())
        all_initial_states = list(loaded_data[all_tasks[0]].keys())
        all_average_task_deviations = []
        all_average_task_costs = []
        for task in all_tasks:
            task_specific_deviations = []
            task_specific_costs = []
            for initial_state in all_initial_states:
                #observations = loaded_data[task][initial_state][OBSERVATIONS_LOG_KEY]
                #control_means = loaded_data[task][initial_state][CONTROL_MEANS_LOG_KEY]
                control_deviations = loaded_data[task][initial_state][CONTROL_DEVIATIONS_LOG_KEY]
                control_costs = loaded_data[task][initial_state][CONTROL_COSTS_LOG_KEY]
                task_specific_deviations.append(control_deviations.squeeze())
                task_specific_costs.append(control_costs.squeeze())
            task_specific_deviations = np.stack(task_specific_deviations)
            task_specific_costs = np.stack(task_specific_costs)
            all_average_task_deviations.append(np.mean(np.sum(task_specific_deviations, axis=1)))
            all_average_task_costs.append(np.mean(np.sum(task_specific_costs, axis=1)))
        ax_1.plot(all_tasks, all_average_task_deviations, label=configuration_label, color=matplotlibcolors[iterator])
        ax_2.plot(all_tasks, all_average_task_costs, label=configuration_label, color=matplotlibcolors[iterator])
    ax_1.set_xlabel('Tasks')
    ax_1.set_xticks(np.linspace(1., 100., 10))
    ax_1.set_ylabel('Standard Deviation')
    #ax_1.set_yticks(np.arange(0., 3.0, 0.2))
    ax_1.set_title('Uncertainty on tasks')
    ax_1.legend()
    ax_2.set_xlabel('Tasks')
    ax_2.set_xticks(np.linspace(1., 100., 10))
    ax_2.set_ylabel('Episodic Cost')
    #ax_2.set_yticks(np.arange(0., 3.0, 0.2))
    ax_2.set_title('Episodic costs on tasks')
    ax_2.legend()
    plt.show()

    '''
    barlist = plt.bar(ALL_BLOCK_MASSES_TO_VALIDATE, uncertainties, label='Unseen Tasks')
    #barlist[9].set_color('r')
    for marker in markers:
        barlist[int(marker)].set_color('r')
    plt.plot([],[], color='r', label='Seen Tasks')
    plt.xlabel('Block Masses')
    plt.ylabel('Standard Deviation')
    #plt.title('Trained context is ' + str(identifier))
    plt.legend()
    plt.show()
    #plt.savefig(expert_file_name)
    #plt.close('all')
    '''