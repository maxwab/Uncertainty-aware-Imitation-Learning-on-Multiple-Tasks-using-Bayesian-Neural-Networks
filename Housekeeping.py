import numpy as np


############## Specific to MuJoCo tasks only #################################
NUMBER_VALIDATION_TRIALS = 10
EXPERT_EPISODES_TO_LOG = 1000

VIDEO_LOGS_DIRECTORY = 'video_logs/'

EXPERT_TRAJECTORY_KEY = 'expert_trajectories_key'
SCALE_KEY = 'scale_key'
OFFSET_KEY = 'offset_key'

EXPERT_OBSERVATIONS_KEY = 'expert_observations_key'
EXPERT_ACTIONS_KEY = 'expert_actions_key'
EXPERT_REWARDS_KEY = 'expert_rewards_key'
EXPERT_UNSCALED_OBSERVATIONS_KEY = 'expert_unscaled_observations_key'
##############################################################################


############## Specific to sliding block experiment only #####################
MAXIMUM_NUMBER_OF_STEPS  = 20

ALL_BLOCK_MASSES_TO_VALIDATE = np.linspace(1., 100., 100)
INITIALIZATION_STATES_TO_VALIDATE = np.array([[-5., -5.], [5., -5.], [2.5, -2.5], [-2.5, 2.5], [-5., 5.], [5., 5.]])
##############################################################################

#Defining colors for highlighting important aspects
GREEN = lambda x: '\x1b[32m{}\x1b[0m'.format(x)
BLUE = lambda x: '\x1b[34m{}\x1b[0m'.format(x)
RED = lambda x: '\x1b[31m{}\x1b[0m'.format(x)


INPUT_MANIPULATION_DIRECTORY = 'input_manipulation_directory/'
TENSORBOARD_DIRECTORY = 'tensorboard_directory/'
SAVED_MODELS_DURING_ITERATIONS_DIRECTORY = 'saved_models_during_iterations/'
SAVED_FINAL_MODEL_DIRECTORY = 'saved_final_model/'
LOGS_DIRECTORY = 'logs/'

MEAN_KEY_X = 'mean_key_x'
DEVIATION_KEY_X = 'deviation_key_x'
MEAN_KEY_Y = 'mean_key_y'
DEVIATION_KEY_Y = 'deviation_key_y'
DRIFT_PER_TIME_STEP_KEY = 'drift_per_time_step_key'
MOVING_WINDOWS_X_SIZE_KEY = 'moving_windows_x_size_key'

UNOPTIMIZED_GP_FIT_KEY = 'unoptimized_GP_fit_key'
OPTIMIZED_GP_FIT_KEY = 'optimized_GP_fit_key'
UNOPTIMIZED_GP_TRAINABLES_KEY = 'unoptimized_GP_trainables_key'
OPTIMIZED_GP_TRAINABLES_KEY = 'optimized_GP_trainables_key'
MEAN_GP_FIT_PREDICTIVE_ERROR_KEY = 'mean_GP_fit_predictive_error_key'
MEAN_GP_FIT_VARIANCE_KEY = 'mean_GP_fit_variance_key'

EXPERIMENT_ID_KEY = 'experiment_id_key'
BEHAVIORAL_CONTROLLER_KEY = 'behavioral_controller_key'
TARGET_CONTROLLER_KEY = 'target_controller_key'
CONTEXTS_KEY = 'contexts_key'
CONTEXT_CODE_KEY = 'context_code_key'
WINDOW_SIZE_KEY = 'window_size_key'
PARTIAL_OBSERVABILITY_KEY = 'partial_observability_key'

OBSERVATIONS_LOG_KEY = 'observations_log_key'

DEMONSTRATOR_CONTROLS_LOG_KEY = 'demonstrator_controls_log_key'
DEMONSTRATOR_COSTS_LOG_KEY = 'demonstrator_costs_log_key'

BEHAVIORAL_CONTROL_COSTS_LOG_KEY = 'behavioral_control_costs_log_key'
BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY = 'behavioral_control_deviations_log_key'
BEHAVIORAL_CONTROL_MEANS_LOG_KEY = 'behavioral_control_means_log_key'
BEHAVIORAL_CONTROL_MAXIMUMS_LOG_KEY = 'behavioral_control_maximums_log_key'
BEHAVIORAL_CONTROL_MINIMUMS_LOG_KEY = 'behavioral_control_minimums_log_key'

TARGET_CONTROL_COSTS_LOG_KEY = 'target_control_costs_log_key'
TARGET_CONTROL_DEVIATIONS_LOG_KEY = 'target_control_deviations_log_key'
TARGET_CONTROL_MEANS_LOG_KEY = 'target_control_means_log_key'

matplotlibcolors = ['black', 'red', 'sienna', 'sandybrown', 'gold', 'olivedrab', 'deepskyblue', 'blue', 'red', 'chartreuse', 'darkcyan']
barwidth = 0.25

##############################################################

MAXIMUM_ACTION_LOG_KEY = 'maximum_action_log_key'
MINIMUM_ACTION_LOG_KEY = 'minimum_action_log_key'
POSITION_GAIN_KEY = 'position_gain_key'
VELOCITY_GAIN_KEY = 'velocity_gain_key'


NORMALIZE = lambda data, mean, deviation: np.divide(np.subtract(data, mean), deviation)
REVERSE_NORMALIZE = lambda data, mean, deviation: np.add((data * deviation), mean)


def get_mean_and_deviation(data):
  mean_data = np.mean(data, axis = 0)
  deviation_data = np.std(data, axis = 0)
  for feature_index in range(deviation_data.shape[0]):
    if deviation_data[feature_index] == 0.:
      if mean_data[feature_index] == 0.:
        # This means all the values are 0.
        deviation_data[feature_index] = 1.
      else:
        # This means all the values are equal but not equal to 0.
        deviation_data[feature_index] = mean_data[feature_index]
  return mean_data, deviation_data


def get_sliding_block_context_from_code(context_code):
  if context_code == 0:
    contexts = [10.]
  elif context_code == 1:
    contexts = [25.]
  elif context_code == 2:
    contexts = [50.]
  elif context_code == 3:
    contexts = [65.]
  elif context_code == 4:
    contexts = [80.]
  elif context_code == 5:
    contexts = [80., 85.]
  elif context_code == 6:
    contexts = [5., 10.]
  elif context_code == 7:
    contexts = [40., 60.]
  else: 
    contexts = [10., 90.]
  return contexts


def get_states_grid(resolution=11):
    all_states, all_velocities = np.meshgrid(np.linspace(-5, 5, resolution), np.linspace(-5, 5, resolution))
    all_states = np.reshape(all_states, (-1, 1))
    all_velocities = np.reshape(all_velocities, (-1, 1))
    states_grid = np.append(all_states, all_velocities, axis=1)
    return states_grid


def get_code_from_sliding_block_context(context):
  pass


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
