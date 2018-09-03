import numpy as np
import _pickle as pickle
import gpflow
import argparse
import os
from datetime import datetime
from scipy.spatial.distance import cdist

from Dataset import getDemonstrationDataset
from Sliding_Block import *
from Validate_GP_Controller import validate_GP_controller

import sys
sys.path.insert(0,'./../Task_Agnostic_Online_Multitask_Imitation_Learning/')

from Housekeeping import *

def generate_GP_controller(contexts, window_size, partial_observability, behavior_controller, target_controller):
    start_time = datetime.now()
    moving_windows_x, moving_windows_y, drift_per_time_step, moving_windows_x_size = getDemonstrationDataset(all_block_masses=contexts,
                                                         window_size=window_size,
                                                         partial_observability=partial_observability)
    print(RED('Time taken to generate dataset is ' + str(datetime.now()-start_time)))
    
    '''
    print(GREEN('Heuristic values of the parameters'))
    kernel_variance = np.var(moving_windows_y)
    kernel_lengthscales = np.median(cdist(moving_windows_x, moving_windows_x, 'sqeuclidean').flatten())
    print(BLUE('Kernel Variance is ' + str(kernel_variance)))
    print(BLUE('Kernel lengthscales is ' + str(kernel_lengthscales)))
    print(BLUE('Likelihood variance is 1/%-10/% /of ' + str(kernel_variance)))
    '''

    k = gpflow.kernels.RBF(1, lengthscales=0.3)
    #k = gpflow.kernels.Matern52(1, lengthscales=0.3)
    #meanf = gpflow.mean_functions.Linear(1.0, 0.0)
    #m = gpflow.models.GPR(moving_windows_x, moving_windows_y, k, meanf)
    m = gpflow.models.GPR(moving_windows_x, moving_windows_y, k)
    m.likelihood.variance = 0.01
    #print(m.read_trainables())
    #print(m.as_pandas_table())
    
    start_time = datetime.now()
    gpflow.train.ScipyOptimizer().minimize(m)
    print(RED('Time taken to optimize the parameters is ' + str(datetime.now()-start_time)))

    #plot(m)
    #print(m.read_trainables())
    #print(m.as_pandas_table())

    #print(m.kern.lengthscales.read_value())

    if behavior_controller == 'GP': behavior_controller = m
    if target_controller == 'GP': target_controller = m

    start_time = datetime.now()
    validate_GP_controller(contexts=contexts, window_size=window_size, partial_observability=partial_observability, drift_per_time_step=drift_per_time_step, moving_windows_x_size=moving_windows_x_size, behavior_controller=behavior_controller, target_controller=target_controller)
    print(RED('Time taken for the validation step is ' + str(datetime.now()-start_time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--contexts', type=int, help='Contexts to train on', default=0)
    parser.add_argument('-ws', '--window_size', type=int, help='Number of time-steps in a moving window', default=1)
    parser.add_argument('-po', '--partial_observability', type=str, help='Partial Observability', default='True')
    parser.add_argument('-bc', '--behavior_controller', type=str, help='Behavior Controller', default='GP', choices=['GP', 'LQR'])
    parser.add_argument('-tc', '--target_controller', type=str, help='Target Controller', default='LQR', choices=['GP', 'LQR'])
    args = parser.parse_args()
    if args.contexts == 0:
        contexts = [10.]
    elif args.contexts == 1:
        contexts = [25.]
    elif args.contexts == 2:
        contexts = [50.]
    elif args.contexts == 3:
        contexts = [65.]
    else:
        contexts = [80.]

    print(GREEN('Settings are contexts ' + str(contexts) + ', window size is ' + str(args.window_size) + ', partial observability is ' + str(args.partial_observability)))

    generate_GP_controller(contexts=contexts, window_size=args.window_size, partial_observability=str_to_bool(args.partial_observability), behavior_controller=args.behavior_controller, target_controller=args.target_controller)
