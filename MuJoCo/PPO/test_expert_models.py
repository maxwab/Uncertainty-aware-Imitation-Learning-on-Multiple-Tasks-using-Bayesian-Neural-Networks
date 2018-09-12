import tensorflow as tf
import argparse, gym, sys, os
import _pickle as pickle
import numpy as np

sys.path.insert(0, '../utilities/')
from general_settings import *
from tweak_context import get_contextual_MUJOCO_environment
from plots import plotExperts

#Defining colors for highlighting important aspects
GREEN = lambda x: '\x1b[32m{}\x1b[0m'.format(x)
BLUE = lambda x: '\x1b[34m{}\x1b[0m'.format(x)
RED = lambda x: '\x1b[31m{}\x1b[0m'.format(x)


def getScaleAndOffset(env_name, expert_context):
	FILE_NAME = EXPERT_TRAJECTORIES_DIRECTORY + env_name + '_' + expert_context + '.pkl'
	with open(FILE_NAME, 'rb') as f:
		all_stored_data = pickle.load(f)
	scale = all_stored_data[SCALE_KEY]
	offset = all_stored_data[OFFSET_KEY]
	return scale, offset


def load_trained_model(env_name):

	reward_on_itself = []

	for expert_context in ALL_CONTEXTS:
		expert_context = str(expert_context)
		scale, offset = getScaleAndOffset(env_name, expert_context)

		saved_final_model = SAVED_FINAL_MODEL_DIRECTORY_EXPERT + env_name + '/' + expert_context + '/'
		tf.reset_default_graph()
		imported_meta = tf.train.import_meta_graph(saved_final_model + 'final.meta')

		with tf.Session() as sess:  
			imported_meta.restore(sess, tf.train.latest_checkpoint(saved_final_model))
			graph = tf.get_default_graph()

			scaled_observation_node = graph.get_tensor_by_name('obs:0')
			output_action_node = graph.get_tensor_by_name('output_action:0')

			expert_rewards_over_all_contexts = []
			for validation_context in ALL_CONTEXTS:
				validation_context = str(validation_context)
				total_reward_over_all_episodes = 0.
				env = get_contextual_MUJOCO_environment(env_name=env_name, context=validation_context)
				#env = wrappers.Monitor(env, "./recordings/" + env_name, force=True)
				observation = env.reset()
				for episode_iterator in range(NUMBER_VALIDATION_TRIALS):
					#print(GREEN('Episode number to validate ' + str(episode_iterator)))
					total_reward = 0.
					finish = False
					time_step = 0.

					while not finish:
						#env.render()
						observation = observation.astype(np.float32).reshape((1, -1))
						observation = np.append(observation, [[time_step]], axis=1)  # add time step feature
						output_action = sess.run(output_action_node, feed_dict={scaled_observation_node:(observation - offset) * scale})
						observation, reward, finish, info = env.step(output_action)
						total_reward += reward
						time_step += 1e-3
						if finish:
							observation = env.reset()
					total_reward_over_all_episodes += total_reward
					#print(RED('Reward obtained during this episode is ' + str(total_reward)))

				#print(RED('Average expert reward is ' + str(total_reward_over_all_episodes/NUMBER_VALIDATION_TRIALS)))
				if expert_context == validation_context:
					reward_on_itself.append(total_reward_over_all_episodes/NUMBER_VALIDATION_TRIALS)
				expert_rewards_over_all_contexts.append(total_reward_over_all_episodes/NUMBER_VALIDATION_TRIALS)
				env.close()

			plotExperts(env_name=env_name, expert_rewards_over_all_contexts=expert_rewards_over_all_contexts, identifier=str(expert_context))

	file_name_to_log_data = EXPERT_CONTROLLER_REWARD_LOG_DIRECTORY + env_name + '.pkl'
	with open(file_name_to_log_data, "wb") as f:
		pickle.dump(reward_on_itself, f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=('Reload and reuse policy trained using PPO '
												  ' on OpenAI Gym environment'))
	parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')

	args = parser.parse_args()
	load_trained_model(**vars(args))
