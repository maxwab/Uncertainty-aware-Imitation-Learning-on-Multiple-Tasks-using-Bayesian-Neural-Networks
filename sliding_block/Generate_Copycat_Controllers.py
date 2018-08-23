from BayesianNNRegression import BBBNNRegression
from Expert_Trajectories import getExpertDataset
from Settings import *
from Utils import *

from datetime import datetime
import tensorflow as tf, argparse, sys, os, copy, _pickle as pickle


def runSimulation(env_name, visibility, case, window_size, epochs, number_mini_batches, activation_unit, learning_rate, hidden_units, number_samples_variance_reduction, precision_alpha, weights_prior_mean_1,
					 weights_prior_mean_2, weights_prior_deviation_1, weights_prior_deviation_2, mixture_pie, rho_mean, extra_likelihood_emphasis):

	directory_to_save_input_manipulation_data = INPUT_MANIPULATION_DIRECTORY + env_name + '/'
	if not os.path.exists(directory_to_save_input_manipulation_data):
		os.makedirs(directory_to_save_input_manipulation_data)
	
	file_name_to_save_input_manipulation_data = directory_to_save_input_manipulation_data + str(visibility) + '_' + str(case) + '.pkl'

	directory_to_save_tensorboard_data = TENSORBOARD_DIRECTORY + env_name + '/' + str(visibility) + '/' + str(case) + '/'
	if not os.path.exists(directory_to_save_tensorboard_data):
		os.makedirs(directory_to_save_tensorboard_data)

	saved_models_during_iterations_copycat = SAVED_MODELS_DURING_ITERATIONS_DIRECTORY_COPYCAT + env_name + '/' + str(visibility) + '/' + str(case) + '/'
	saved_final_model_copycat = SAVED_FINAL_MODEL_DIRECTORY_COPYCAT + env_name + '/' + str(visibility) + '/' + str(case) + '/'

	if not os.path.exists(saved_models_during_iterations_copycat):
		os.makedirs(saved_models_during_iterations_copycat)
	if not os.path.exists(saved_final_model_copycat):
		os.makedirs(saved_final_model_copycat)

	if case == 0:
		block_masses_to_train_on = BLOCK_MASSES_TO_TRAIN_ON_1
	elif case == 1:
		block_masses_to_train_on = BLOCK_MASSES_TO_TRAIN_ON_2
	elif case == 2:
		block_masses_to_train_on = BLOCK_MASSES_TO_TRAIN_ON_3
	else:
		print('Case specified is not valid')
		exit(0)

	x_seen, y_seen, observation_dimensions_per_time_step = getExpertDataset(all_block_masses=block_masses_to_train_on, window_size=window_size)

	if not visibility:
		x_seen = np.delete(x_seen, np.s_[2::5], 1)
		observation_dimensions_per_time_step = observation_dimensions_per_time_step - 1

	# House-keeping to make data amenable for good training
	mean_x = np.mean(x_seen, axis = 0)
	deviation_x = np.std(x_seen, axis = 0)
	x_seen = NORMALIZE(x_seen, mean_x, deviation_x)

	mean_y = np.mean(y_seen, axis = 0)
	deviation_y = np.std(y_seen, axis = 0)
	y_seen = NORMALIZE(y_seen, mean_y, deviation_y)

	normalization_data_to_store = {MEAN_KEY_X: mean_x, DEVIATION_KEY_X: deviation_x, MEAN_KEY_Y:mean_y, DEVIATION_KEY_Y:deviation_y,
									 OBSERVATION_DIMENSIONS_PER_TIME_STEP_KEY: observation_dimensions_per_time_step,
									 	 OBSERVATION_WINDOW_SIZE_KEY: window_size}

	with open(file_name_to_save_input_manipulation_data, 'wb') as f:
		pickle.dump(normalization_data_to_store, f)

	
	print(GREEN('Creating the BBB based Bayesian NN'))
	BBB_Regressor=BBBNNRegression(number_mini_batches=number_mini_batches, number_features=x_seen.shape[1], number_output_units=y_seen.shape[1], activation_unit=activation_unit, learning_rate=learning_rate,
										 hidden_units=hidden_units, number_samples_variance_reduction=number_samples_variance_reduction, precision_alpha=precision_alpha,
											 weights_prior_mean_1=weights_prior_mean_1, weights_prior_mean_2=weights_prior_mean_2, weights_prior_deviation_1=weights_prior_deviation_1,
												 weights_prior_deviation_2=weights_prior_deviation_2, mixture_pie=mixture_pie, rho_mean=rho_mean, extra_likelihood_emphasis=extra_likelihood_emphasis)
	print(GREEN('BBB based Bayesian NN created successfully'))

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter(directory_to_save_tensorboard_data, sess.graph)
		saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=2)
		previous_minimum_loss = sys.float_info.max

		mini_batch_size = int(x_seen.shape[0]/number_mini_batches)

		for epoch_iterator in range(epochs):
			x_seen, y_seen = randomize(x_seen, y_seen)
			ptr = 0
			for mini_batch_iterator in range(number_mini_batches):
				x_batch = x_seen[ptr:ptr+mini_batch_size, :]
				y_batch = y_seen[ptr:ptr+mini_batch_size, :]

				_, loss, summary = sess.run([BBB_Regressor.train(), BBB_Regressor.getMeanSquaredError(), BBB_Regressor.summarize()], feed_dict={BBB_Regressor.X_input:x_batch, BBB_Regressor.Y_input:y_batch})
				sess.run(BBB_Regressor.update_mini_batch_index())
				
				if loss < previous_minimum_loss:
					saver.save(sess, saved_models_during_iterations_copycat + 'iteration', global_step=epoch_iterator, write_meta_graph=False)
					previous_minimum_loss = loss
				
				ptr += mini_batch_size
				writer.add_summary(summary, global_step=tf.train.global_step(sess, BBB_Regressor.global_step))
			
			if epoch_iterator % 4000 == 0:
				print(RED('Training progress: ' + str(epoch_iterator) + '/' + str(epochs)))
		
		writer.close()
		saver.save(sess, saved_final_model_copycat + 'final', write_state=False)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--env', type=str, help='Name ID environment to run', default='SlidingBlock')
	parser.add_argument('-v', '--visibility', type=str, help='Context Visibility', default='False')
	parser.add_argument('-c', '--case', type=int, help='Case', default=2)
	args = parser.parse_args()

	start_time = datetime.now()
	runSimulation(env_name=args.env, visibility=str_to_bool(args.visibility), case=args.case, window_size=2, epochs = 10001, number_mini_batches = 20, activation_unit = 'RELU', learning_rate = 0.001, hidden_units= [90, 30, 10],
					 number_samples_variance_reduction = 25, precision_alpha = 0.01, weights_prior_mean_1 = 0., weights_prior_mean_2 = 0., weights_prior_deviation_1 = 0.4, weights_prior_deviation_2 = 0.4,
					 	 mixture_pie = 0.7, rho_mean = -3., extra_likelihood_emphasis = 10000000000000000.)	
	#GoGoGo(env_name=args.env, exploration_type=args.st, window_size=2, epochs = 20001, number_mini_batches = 20, activation_unit = 'RELU', learning_rate = 0.001,
	#		 hidden_units= [90, 30, 10], number_samples_variance_reduction = 25, precision_alpha = 0.01, weights_prior_mean_1 = 0., weights_prior_mean_2 = 0., weights_prior_deviation_1 = 0.4,
	#			 weights_prior_deviation_2 = 0.4, mixture_pie = 0.7, rho_mean = -3., extra_likelihood_emphasis = 10000000000000000., number_episodes_to_test=10, filter_settings=args.s)
	print('Total time taken is ' + str(datetime.now() - start_time))