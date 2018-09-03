import tensorflow as tf, argparse, _pickle as pickle, os
from Load_Trained_Controllers import Load_Copycat
from Sliding_Block import *
from LQR import *
from Settings import *
from Utils import *


def validate_BBB_controller(env_name, visibility, case):
	copycat_graph = tf.Graph()
	with copycat_graph.as_default():
		copycat_controller = Load_Copycat(env_name=env_name, visibility=visibility, case=case)

	logs_directory = LOGS_DIRECTORY + env_name + '/'
	if not os.path.exists(logs_directory):
		os.makedirs(logs_directory)

	logs_file = logs_directory + str(visibility) + '_' + str(case) + '.pkl'

	all_Ks = []

	logs_for_all_block_masses = {}

	for block_mass in ALL_BLOCK_MASSES_TO_VALIDATE:
		#cost_at_this_mass = 0
		#uncertainty_at_this_mass = 0
		#total_steps_at_this_mass = 0

		flag_for_K_vs_Block_Mass = 0

		logs_for_all_initial_states = {}

		for initial_state in INITIALIZATION_STATES:
			logs_related_to_the_current_initial_state = {}
			cost_incurred_during_all_time_steps = []
			action_taken_at_all_time_steps = []
			deviation_over_dynamics_at_all_time_steps = []
			minimum_of_actions_at_all_time_steps = []
			maximum_of_actions_at_all_time_steps = []
			
			env = Sliding_Block(mass=block_mass, initial_state=initial_state)

			K, X, eigVals = dlqr(env.A, env.B, env.Q, env.R)

			if flag_for_K_vs_Block_Mass == 0:
				all_Ks.append(K[0])
				flag_for_K_vs_Block_Mass = 1

			observation = env.state
			finish = False

			input_validation = np.full((1, (copycat_controller.observation_dimensions_per_time_step*copycat_controller.observation_window_size)-2), 0.)

			if not visibility == True:
				input_validation[0, -(observation.shape[0]):] = observation.T[0]
			else:	
				input_validation[0, -(observation.shape[0]+1):-1] = observation.T[0]
				input_validation[0, -1] = block_mass

			action_this_time_step = (-1. * np.dot(K, observation))[0,0]
			#action_this_time_step = np.random.uniform(low=FIRST_ACTION_LOW, high=FIRST_ACTION_HIGH, size=(1,1))
			#action_this_time_step = np.array([[5.]])

			observation, cost, finish = env.step(action_this_time_step)

			'''
			if not visibility == True:
				if not copycat_controller.observation_window_size == 1:
					input_validation[0, :-4] = input_validation[0, 4:]
					input_validation[0,-4] = action_this_time_step
					input_validation[0,-3] = -cost
				input_validation[0, -(observation.shape[0]):] = observation.T[0]
			else:
				if not copycat_controller.observation_window_size == 1:
					input_validation[0, :-5] = input_validation[0, 5:]
					input_validation[0,-5] = action_this_time_step
					input_validation[0,-4] = -cost
				input_validation[0, -(observation.shape[0]+1):-1] = observation.T[0]
				input_validation[0, -1] = block_mass


			action_this_time_step, deviation_this_time_step, maximum_this_time_step, minimum_this_time_step = copycat_controller.sess.run([copycat_controller.mean_of_predictions, copycat_controller.deviation_of_predictions, copycat_controller.maximum_of_predictions, copycat_controller.minimum_of_predictions], feed_dict={copycat_controller.x_input:NORMALIZE(input_validation, copycat_controller.mean_x, copycat_controller.deviation_x)})
			action_this_time_step = REVERSE_NORMALIZE(action_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
			maximum_this_time_step = REVERSE_NORMALIZE(maximum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
			minimum_this_time_step = REVERSE_NORMALIZE(minimum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)

			deviation_this_time_step = deviation_this_time_step * copycat_controller.deviation_y
			'''

			#copycat_action, copycat_uncertainty, copycat_maximum, copycat_minimum = sess.run(BBB_Regressor.makeInference(), feed_dict={BBB_Regressor.X_input: NORMALIZE(input_validation, mean_x, deviation_x)})
			#copycat_action = REVERSE_NORMALIZE(copycat_action, mean_y, deviation_y)
			#copycat_uncertainty = REVERSE_NORMALIZE(copycat_uncertainty, mean_y, deviation_y)

			step_limit = 0
			while (step_limit < MAXIMUM_NUMBER_OF_STEPS):

				'''
				#Limiting the position domain
				if observation[0, 0] < -100.:
					observation[0, 0] = -100.
				if observation[0, 0] > 100.:
					observation[0, 0] = 100.

				#Limiting the velocity domain
				if observation[1, 0] < -30.:
					observation[1, 0] = -30.
				if observation[1, 0] > 30.:
					observation[1, 0] = 30.
				'''

				if not visibility == True:
					if not copycat_controller.observation_window_size == 1:
						input_validation[0, :-4] = input_validation[0, 4:]
						input_validation[0,-4] = action_this_time_step
						input_validation[0,-3] = -cost
					input_validation[0, -(observation.shape[0]):] = observation.T[0]
				else:
					if not copycat_controller.observation_window_size == 1:
						input_validation[0, :-5] = input_validation[0, 5:]
						input_validation[0,-5] = action_this_time_step
						input_validation[0,-4] = -cost
					input_validation[0, -(observation.shape[0]+1):-1] = observation.T[0]
					input_validation[0, -1] = block_mass

				action_this_time_step, deviation_this_time_step, maximum_this_time_step, minimum_this_time_step = copycat_controller.sess.run([copycat_controller.mean_of_predictions, copycat_controller.deviation_of_predictions, copycat_controller.maximum_of_predictions, copycat_controller.minimum_of_predictions], feed_dict={copycat_controller.x_input:NORMALIZE(input_validation, copycat_controller.mean_x, copycat_controller.deviation_x)})
				action_this_time_step = REVERSE_NORMALIZE(action_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
				maximum_this_time_step = REVERSE_NORMALIZE(maximum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)
				minimum_this_time_step = REVERSE_NORMALIZE(minimum_this_time_step, copycat_controller.mean_y, copycat_controller.deviation_y)

				deviation_this_time_step = deviation_this_time_step * copycat_controller.deviation_y

				#copycat_action, copycat_uncertainty, copycat_maximum, copycat_minimum = sess.run(BBB_Regressor.makeInference(), feed_dict={BBB_Regressor.X_input:NORMALIZE(input_validation, mean_x, deviation_x)})
				#copycat_action = REVERSE_NORMALIZE(copycat_action, mean_y, deviation_y)
				#copycat_uncertainty = REVERSE_NORMALIZE(copycat_uncertainty, mean_y, deviation_y)

				#cost_at_this_state += cost[0,0]
				#uncertainty_at_this_state += deviation_this_time_step[0,0]

				step_limit += 1					
				observation, cost, finish = env.step(action_this_time_step)

				cost_incurred_during_all_time_steps.append(-cost[0, 0])
				action_taken_at_all_time_steps.append(action_this_time_step[0, 0])
				deviation_over_dynamics_at_all_time_steps.append(deviation_this_time_step[0, 0])
				minimum_of_actions_at_all_time_steps.append(minimum_this_time_step[0, 0])
				maximum_of_actions_at_all_time_steps.append(maximum_this_time_step[0, 0])

			logs_related_to_the_current_initial_state[COST_LOG_KEY] = cost_incurred_during_all_time_steps
			logs_related_to_the_current_initial_state[DEVIATION_LOG_KEY] = deviation_over_dynamics_at_all_time_steps
			logs_related_to_the_current_initial_state[ACTION_TAKEN_LOG_KEY] = action_taken_at_all_time_steps
			logs_related_to_the_current_initial_state[MAXIMUM_ACTION_LOG_KEY] = maximum_of_actions_at_all_time_steps
			logs_related_to_the_current_initial_state[MINIMUM_ACTION_LOG_KEY] = minimum_of_actions_at_all_time_steps


			logs_for_all_initial_states[str(initial_state)] = logs_related_to_the_current_initial_state
			#cost_at_this_mass += (cost_at_this_state/step_limit)
			#uncertainty_at_this_mass += (uncertainty_at_this_state/step_limit)
			#total_steps_at_this_mass += step_limit

		logs_for_all_initial_states[POSITION_GAIN_KEY] = K[0, 0]
		logs_for_all_initial_states[VELOCITY_GAIN_KEY] = K[0, 1]
		logs_for_all_block_masses[str(block_mass)] = logs_for_all_initial_states
		#cost_over_all_block_masses.append(cost_at_this_mass/INITIALIZATION_STATES.shape[0])
		#uncertainties_over_all_block_masses.append(uncertainty_at_this_mass/INITIALIZATION_STATES.shape[0])
		#total_steps_taken_over_all_block_masses.append(total_steps_at_this_mass)

	with open(logs_file, 'wb') as f:
		pickle.dump(logs_for_all_block_masses, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--contexts', type=int, help='Contexts to train on', default=0)
    parser.add_argument('-ws', '--window_size', type=int, help='Number of time-steps in a moving window', default=1)
    parser.add_argument('-po', '--partial_observability', type=str, help='Partial Observability', default='True')
    parser.add_argument('-bc', '--behavior_controller', type=str, help='Behavior Controller', default='BBB', choices=['BBB', 'LQR'])
    parser.add_argument('-tc', '--target_controller', type=str, help='Target Controller', default='LQR', choices=['BBB', 'LQR'])
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