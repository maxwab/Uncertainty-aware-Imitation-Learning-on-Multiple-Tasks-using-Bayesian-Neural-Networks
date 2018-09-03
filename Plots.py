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
    ax_1 = fig.add_subplot(221, frameon=False)
    ax_2 = fig.add_subplot(222, frameon=False)
    ax_3 = fig.add_subplot(223, frameon=False)
    for iterator, configuration in enumerate(all_task_configurations):
        file_to_load_data_from = './' + configuration[EXPERIMENT_ID_KEY] + '/' + LOGS_DIRECTORY + configuration[CONTEXTS_KEY] + '_' + configuration[WINDOW_SIZE_KEY] + '_' + configuration[PARTIAL_OBSERVABILITY_KEY] + '_' + configuration[BEHAVIORAL_CONTROLLER_KEY] + '_' + configuration[TARGET_CONTROLLER_KEY] + '.pkl'
        configuration_label = configuration[CONTEXTS_KEY] + '_' + configuration[WINDOW_SIZE_KEY] + '_' + configuration[PARTIAL_OBSERVABILITY_KEY] + '_' + configuration[BEHAVIORAL_CONTROLLER_KEY] + '_' + configuration[TARGET_CONTROLLER_KEY]
        with open(file_to_load_data_from, 'rb') as f:
            loaded_data = pickle.load(f)
        all_tasks = list(loaded_data.keys())
        all_initial_states = list(loaded_data[all_tasks[0]].keys())
        behavioral_average_task_deviations = []
        behavioral_average_task_costs = []
        average_predictive_error = []

        for task in all_tasks:
            behavioral_task_deviations = []
            behavioral_task_costs = []
            predictive_error = []

            for initial_state in all_initial_states:
                #observations = loaded_data[task][initial_state][OBSERVATIONS_LOG_KEY]
                behavioral_control_means = loaded_data[task][initial_state][BEHAVIORAL_CONTROL_MEANS_LOG_KEY]
                target_control_means = loaded_data[task][initial_state][TARGET_CONTROL_MEANS_LOG_KEY]
                behavioral_control_deviations = loaded_data[task][initial_state][BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY]
                behavioral_control_costs = loaded_data[task][initial_state][BEHAVIORAL_CONTROL_COSTS_LOG_KEY]
                behavioral_task_deviations.append(behavioral_control_deviations.squeeze())
                behavioral_task_costs.append(behavioral_control_costs.squeeze())
                predictive_error.append(np.mean(np.square(behavioral_control_means-target_control_means)))

            behavioral_task_deviations = np.stack(behavioral_task_deviations)
            behavioral_task_costs = np.stack(behavioral_task_costs)
            behavioral_average_task_deviations.append(np.mean(np.sum(behavioral_task_deviations, axis=1)))
            behavioral_average_task_costs.append(np.mean(np.sum(behavioral_task_costs, axis=1)))
            average_predictive_error.append(np.mean(predictive_error))
        ax_1.plot(all_tasks, behavioral_average_task_deviations, label='Behavior_'+configuration_label, color=matplotlibcolors[iterator])
        ax_2.plot(all_tasks, behavioral_average_task_costs, label='Behavior_'+configuration_label, color=matplotlibcolors[iterator])
        ax_3.plot(all_tasks, average_predictive_error, label='Predictive Error', color=matplotlibcolors[iterator])
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
    
    ax_3.set_xlabel('Tasks')
    ax_3.set_xticks(np.linspace(1., 100., 10))
    ax_3.set_ylabel('Predictive Error')
    #ax_3.set_yticks(np.arange(0., 3.0, 0.2))
    ax_3.set_title('Predictive Errors on tasks')
    ax_3.legend()
    
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