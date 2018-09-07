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

def generate_GP_controller(context_code, window_size, partial_observability):
    start_time = datetime.now()
    moving_windows_x, moving_windows_y, drift_per_time_step, moving_windows_x_size = getDemonstrationDataset(all_block_masses=get_sliding_block_context_from_code(context_code=context_code),
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

    k = gpflow.kernels.RBF(moving_windows_x.shape[1], lengthscales=0.1*np.std(moving_windows_x, axis=0))
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

    start_time = datetime.now()
    validate_GP_controller(context_code=context_code, window_size=window_size, partial_observability=partial_observability, drift_per_time_step=drift_per_time_step, moving_windows_x_size=moving_windows_x_size, behavior_controller=m)
    print(RED('Time taken for the validation step is ' + str(datetime.now()-start_time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--context_code', type=int, help='Contexts to train on', default=0)
    parser.add_argument('-ws', '--window_size', type=int, help='Number of time-steps in a moving window', default=1)
    parser.add_argument('-po', '--partial_observability', type=str, help='Partial Observability', default='True')
    args = parser.parse_args()

    print(GREEN('Settings are context code ' + str(args.context_code) + ', window size is ' + str(args.window_size) + ', partial observability is ' + str(args.partial_observability)))

    generate_GP_controller(context_code=args.context_code, window_size=args.window_size, partial_observability=str_to_bool(args.partial_observability))
