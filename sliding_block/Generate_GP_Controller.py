import numpy as np
import _pickle as pickle
import gpflow
import argparse
import os

from Dataset import getDemonstrationDataset
from Sliding_Block import *

import sys
sys.path.insert(0,'./../Task_Agnostic_Online_Multitask_Imitation_Learning/')

from Housekeeping import *

def generate_GP_controller(contexts, window_size, partial_observability):
    moving_windows_x, moving_windows_y, drift_per_time_step, moving_windows_x_size = getDemonstrationDataset(all_block_masses=contexts,
                                                         window_size=window_size,
                                                         partial_observability=partial_observability)
    k = gpflow.kernels.Matern52(1, lengthscales=0.3)
    #meanf = gpflow.mean_functions.Linear(1.0, 0.0)
    #m = gpflow.models.GPR(moving_windows_x, moving_windows_y, k, meanf)
    m = gpflow.models.GPR(moving_windows_x, moving_windows_y, k)
    m.likelihood.variance = 0.01
    #print(m.read_trainables())
    #print(m.as_pandas_table())
    
    gpflow.train.ScipyOptimizer().minimize(m)
    
    #plot(m)
    #print(m.read_trainables())
    #print(m.as_pandas_table())

    #print(m.kern.lengthscales.read_value())

    ############ Validation #############
    if not os.path.exists(LOGS_DIRECTORY):
        os.makedirs(LOGS_DIRECTORY)
    file_to_save_logs = './' + LOGS_DIRECTORY + 'gp_controller_' + str(contexts) + '_' + str(window_size) + '_' + str(partial_observability) + '.pkl'
    logs_for_all_blocks = {}
    for block_mass in ALL_BLOCK_MASSES_TO_VALIDATE:
        logs_for_a_block_and_initial_state = {}
        for initial_state in INITIALIZATION_STATES_TO_VALIDATE:
            all_observations = []
            all_control_means = []
            all_control_deviations = []
            all_costs = []
            env = Sliding_Block(mass=block_mass, initial_state=initial_state)
            total_cost = total_variance = 0.
            observation = env.state
            if not partial_observability:
                observation = np.append(observation.T, np.array([[block_mass]]), axis=1)
            else:
                observation = observation.T
            moving_window_x = np.zeros((1, moving_windows_x_size))
            moving_window_x[0, -observation.shape[1]:] = observation[0]
            mean_control, var_control = m.predict_y(moving_window_x)
            step_limit = 0
            while (step_limit < MAXIMUM_NUMBER_OF_STEPS):      
                step_limit += 1
                all_observations.append(observation)
                all_control_means.append(mean_control)
                all_control_deviations.append(np.sqrt(var_control))
                observation, cost, finish = env.step(mean_control)
                all_costs.append(cost)
                if not window_size == 1:
                    moving_window_x[0, :-drift_per_time_step] = moving_window_x[0, drift_per_time_step:]
                    moving_window_x[0, -drift_per_time_step:-(drift_per_time_step-mean_control.shape[1])] = mean_control[0]
                    moving_window_x[0, -(drift_per_time_step-mean_control.shape[0])] = -cost      
                if not partial_observability:
                    observation = np.append(observation.T, np.array([[block_mass]]), axis=1)
                else:
                    observation = observation.T
                moving_window_x[0, -observation.shape[1]:] = observation[0]
                mean_control, var_control = m.predict_y(moving_window_x)
            logs_for_a_block_and_initial_state[str(initial_state)] = {OBSERVATIONS_LOG_KEY: np.concatenate(all_observations), CONTROL_MEANS_LOG_KEY: np.concatenate(all_control_means),
                                                                     CONTROL_COSTS_LOG_KEY: np.concatenate(all_costs), CONTROL_DEVIATIONS_LOG_KEY: np.concatenate(all_control_deviations)}
        logs_for_all_blocks[str(block_mass)] = logs_for_a_block_and_initial_state
    with open(file_to_save_logs, 'wb') as f:
        pickle.dump(logs_for_all_blocks, f, protocol=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--contexts', type=int, help='Contexts to train on', default=0)
    parser.add_argument('-ws', '--window_size', type=int, help='Number of time-steps in a moving window', default=1)
    parser.add_argument('-po', '--partial_observability', type=str, help='Partial Observability', default='True')
    args = parser.parse_args()
    if args.contexts == 0:
        contexts = [80.]
    elif args.contexts == 1:
        contexts = [10.]
    elif args.contexts == 2:
        contexts = [25.]
    elif args.contexts == 3:
        contexts = [50.]
    else:
        contexts = [65.]
    generate_GP_controller(contexts=contexts, window_size=args.window_size, partial_observability=str_to_bool(args.partial_observability))
