# Don't forget to re-run the expert log and simulator files, if you change any value here.

import numpy as np

INITIALIZATION_STATES = np.array([[-5., -5.], [5., -5.], [2.5, -2.5], [-2.5, 2.5], [-5., 5.], [5., 5.]])
MAXIMUM_NUMBER_OF_STEPS  = 150

BLOCK_MASSES_TO_TRAIN_ON_1 = np.array([2., 3., 4.])
BLOCK_MASSES_TO_TRAIN_ON_2 = np.array([52., 53., 54.])
BLOCK_MASSES_TO_TRAIN_ON_3 = np.array([92., 93., 94.])
ALL_BLOCK_MASSES_TO_VALIDATE = np.linspace(1., 100., 100)

#Defining colors for highlighting important aspects
GREEN = lambda x: '\x1b[32m{}\x1b[0m'.format(x)
BLUE = lambda x: '\x1b[34m{}\x1b[0m'.format(x)
RED = lambda x: '\x1b[31m{}\x1b[0m'.format(x)

INPUT_MANIPULATION_DIRECTORY = './input_manipulation_directory/'
TENSORBOARD_DIRECTORY = './tensorboard_directory/'
SAVED_MODELS_DURING_ITERATIONS_DIRECTORY_COPYCAT = './saved_models_during_iterations_copycat/'
SAVED_FINAL_MODEL_DIRECTORY_COPYCAT = './saved_final_model_copycat/'
LOGS_DIRECTORY = './logs/'

MEAN_KEY_X = 'mean_key_x'
DEVIATION_KEY_X = 'deviation_key_x'
MEAN_KEY_Y = 'mean_key_y'
DEVIATION_KEY_Y = 'deviation_key_y'
OBSERVATION_DIMENSIONS_PER_TIME_STEP_KEY = 'observation_dimensions_per_time_step_key'
OBSERVATION_WINDOW_SIZE_KEY = 'observation_window_size_key'

COST_LOG_KEY = 'cost_log_key'
DEVIATION_LOG_KEY = 'deviation_log_key'
ACTION_TAKEN_LOG_KEY = 'action_taken_log_key'
MAXIMUM_ACTION_LOG_KEY = 'maximum_action_log_key'
MINIMUM_ACTION_LOG_KEY = 'minimum_action_log_key'
POSITION_GAIN_KEY = 'position_gain_key'
VELOCITY_GAIN_KEY = 'velocity_gain_key'


NORMALIZE = lambda data, mean, deviation: np.divide(np.subtract(data, mean), deviation)
REVERSE_NORMALIZE = lambda data, mean, deviation: np.add((data * deviation), mean)


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False


def randomize(a, b):
	# Generate the permutation index array.
	permutation = np.random.permutation(a.shape[0])
	# Shuffle the arrays by giving the permutation in the square brackets.
	shuffled_a = a[permutation]
	shuffled_b = b[permutation]
	return shuffled_a, shuffled_b