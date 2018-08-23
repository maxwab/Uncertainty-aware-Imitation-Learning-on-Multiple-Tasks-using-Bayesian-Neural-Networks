import tensorflow as tf
import _pickle as pickle, sys

from Settings import *

class Load_Copycat():
	def __init__(self, env_name, visibility, case):
		self.sess = tf.Session()

		meta_information_directory_copycat = SAVED_FINAL_MODEL_DIRECTORY_COPYCAT + env_name + '/' + str(visibility) + '/' + str(case) + '/'
		best_model_directory_copycat = SAVED_MODELS_DURING_ITERATIONS_DIRECTORY_COPYCAT + env_name + '/' + str(visibility) + '/' + str(case) + '/'

		imported_meta = tf.train.import_meta_graph(meta_information_directory_copycat + 'final.meta')
		imported_meta.restore(self.sess, tf.train.latest_checkpoint(best_model_directory_copycat))
		graph = tf.get_default_graph()

		self.x_input = graph.get_tensor_by_name('inputs/x_input:0')
		self.y_input = graph.get_tensor_by_name('inputs/y_input:0')
		self.mean_of_predictions = graph.get_tensor_by_name('final_outputs/prediction_mean:0')
		self.deviation_of_predictions = graph.get_tensor_by_name('final_outputs/prediction_standard_deviation:0')
		self.maximum_of_predictions = graph.get_tensor_by_name('final_outputs/prediction_maximum:0')
		self.minimum_of_predictions = graph.get_tensor_by_name('final_outputs/prediction_minimum:0')

		self.getInputManipulationInformation(env_name, visibility, case)

	def getInputManipulationInformation(self, env_name, visibility, case):
		relevant_file_name = INPUT_MANIPULATION_DIRECTORY + env_name + '/' + str(visibility) + '_' + str(case) + '.pkl'
		with open(relevant_file_name, 'rb') as f:
			stored_dataset_manipulation_data = pickle.load(f)
		self.mean_x = stored_dataset_manipulation_data[MEAN_KEY_X]
		self.deviation_x = stored_dataset_manipulation_data[DEVIATION_KEY_X]
		self.mean_y = stored_dataset_manipulation_data[MEAN_KEY_Y]
		self.deviation_y = stored_dataset_manipulation_data[DEVIATION_KEY_Y]
		self.observation_dimensions_per_time_step = stored_dataset_manipulation_data[OBSERVATION_DIMENSIONS_PER_TIME_STEP_KEY]
		self.observation_window_size = stored_dataset_manipulation_data[OBSERVATION_WINDOW_SIZE_KEY]


class Load_Expert():
	def __init__(self, env_name, context):
		self.getScaleAndOffset(env_name, context)
		saved_final_model_expert = SAVED_FINAL_MODEL_DIRECTORY_EXPERT + env_name + '/' + context + '/'
		imported_meta = tf.train.import_meta_graph(saved_final_model_expert + 'final.meta')
		self.sess = tf.Session()
		imported_meta.restore(self.sess, tf.train.latest_checkpoint(saved_final_model_expert))
		graph = tf.get_default_graph()
		self.scaled_observation_node = graph.get_tensor_by_name('obs:0')
		self.output_action_node = graph.get_tensor_by_name('output_action:0')

	def getScaleAndOffset(self, env_name, context):
		file_name = EXPERT_TRAJECTORIES_DIRECTORY + env_name + '_' + context + '.pkl'
		with open(file_name, "rb") as f:
			data_stored = pickle.load(f)
		self.scale = data_stored[SCALE_KEY]
		self.offset = data_stored[OFFSET_KEY]