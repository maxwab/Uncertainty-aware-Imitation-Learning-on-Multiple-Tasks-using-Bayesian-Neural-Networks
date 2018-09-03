import numpy as np

MAXIMUM_NUMBER_OF_STEPS  = 30

BLOCK_MASSES_TO_TRAIN_ON_1 = np.array([2., 3., 4.])
BLOCK_MASSES_TO_TRAIN_ON_2 = np.array([52., 53., 54.])
BLOCK_MASSES_TO_TRAIN_ON_3 = np.array([92., 93., 94.])
ALL_BLOCK_MASSES_TO_VALIDATE = np.linspace(1., 100., 100)
INITIALIZATION_STATES_TO_VALIDATE = np.array([[-5., -5.], [5., -5.], [2.5, -2.5], [-2.5, 2.5], [-5., 5.], [5., 5.]])

#Defining colors for highlighting important aspects
GREEN = lambda x: '\x1b[32m{}\x1b[0m'.format(x)
BLUE = lambda x: '\x1b[34m{}\x1b[0m'.format(x)
RED = lambda x: '\x1b[31m{}\x1b[0m'.format(x)

INPUT_MANIPULATION_DIRECTORY = './input_manipulation_directory/'
TENSORBOARD_DIRECTORY = './tensorboard_directory/'
SAVED_MODELS_DURING_ITERATIONS_DIRECTORY_COPYCAT = './saved_models_during_iterations_copycat/'
SAVED_FINAL_MODEL_DIRECTORY_COPYCAT = './saved_final_model_copycat/'
LOGS_DIRECTORY = 'logs/'

MEAN_KEY_X = 'mean_key_x'
DEVIATION_KEY_X = 'deviation_key_x'
MEAN_KEY_Y = 'mean_key_y'
DEVIATION_KEY_Y = 'deviation_key_y'
OBSERVATION_DIMENSIONS_PER_TIME_STEP_KEY = 'observation_dimensions_per_time_step_key'
OBSERVATION_WINDOW_SIZE_KEY = 'observation_window_size_key'


##############################################################

EXPERIMENT_ID_KEY = 'experiment_id_key'
BEHAVIORAL_CONTROLLER_KEY = 'behavioral_controller_key'
TARGET_CONTROLLER_KEY = 'target_controller_key'
CONTEXTS_KEY = 'contexts_key'
WINDOW_SIZE_KEY = 'window_size_key'
PARTIAL_OBSERVABILITY_KEY = 'partial_observability_key'

OBSERVATIONS_LOG_KEY = 'observations_log_key'

BEHAVIORAL_CONTROL_COSTS_LOG_KEY = 'behavioral_control_costs_log_key'
BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY = 'behavioral_control_deviations_log_key'
BEHAVIORAL_CONTROL_MEANS_LOG_KEY = 'behavioral_control_means_log_key'

TARGET_CONTROL_COSTS_LOG_KEY = 'target_control_costs_log_key'
TARGET_CONTROL_DEVIATIONS_LOG_KEY = 'target_control_deviations_log_key'
TARGET_CONTROL_MEANS_LOG_KEY = 'target_control_means_log_key'

matplotlibcolors = ['black', 'red', 'sienna', 'sandybrown', 'gold', 'olivedrab', 'deepskyblue', 'blue', 'red', 'chartreuse', 'darkcyan']

##############################################################

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


def get_moving_window_size(observation_sample, action_sample, window_size):
    """This function returns the number of dimensions in the moving window feature and target vectors.

    Args:
       observation_sample (2-D numpy array):  A sample observation with just one row.
       action_sample (2-D numpy array): A sample action with just one row
       window size (int): The number of last time-steps in the moving window

    Returns:
       int: drift in terms of the number of dimensions in the moving windows feature vector
       int: number of dimensions in the moving windows feature vector
       int: number of dimensions in the moving windows target vector

    A way you might use me is
    >>> get_moving_window_size(observation_sample=np.array([[1., 2., 3.]]), action_sample=np.array([[10., 5.]]), window_size=3)
    15

    """
    drift_per_time_step = observation_sample.shape[1]+action_sample.shape[1]+1
    moving_window_size_x = (window_size-1)*(drift_per_time_step) + observation_sample.shape[1]
    moving_window_size_y = action_sample.shape[1]
    return drift_per_time_step, moving_window_size_x, moving_window_size_y
