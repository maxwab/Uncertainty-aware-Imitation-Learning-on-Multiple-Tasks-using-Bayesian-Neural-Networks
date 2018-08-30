#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np, os, sys, _pickle as pickle

import sys
#sys.path.insert(0,'./../Task_Agnostic_Online_Multitask_Imitation_Learning/')
sys.path.insert(0,'./../')
from Housekeeping import *


def plot_uncertainty_vs_tasks(all_task_configurations):
	for configuration in all_task_configurations:
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
		plt.plot(all_tasks, all_average_task_deviations, label=configuration_label)
		plt.legend()
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