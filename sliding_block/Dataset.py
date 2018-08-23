import matplotlib.pyplot as plt
import copy, os, sys

from Sliding_Block import *
from LQR import *
from Settings import *


# This function won't work for window size == 1
def getExpertDatasetFromMassNInitialState(mass, initial_state, window_size):
	env = Sliding_Block(mass=mass, initial_state=initial_state)
	observation = env.state
	finish = False

	K, X, eigVals = dlqr(env.A, env.B, env.Q, env.R)

	first_action = (-1. * np.dot(K, observation))[0,0]
	#first_action = np.random.uniform(low=FIRST_ACTION_LOW, high=FIRST_ACTION_HIGH, size=(1,1))
	#first_action = np.array([[5.]])

	trajectories = None
	expert_actions = None

	dataset_window = np.full((1, (window_size*5)-2), 0.)
	dataset_window[0, -(observation.shape[0]+1):-1] = observation.T[0]
	dataset_window[0, -1] = mass

	observation, cost, finish = env.step(first_action)

	dataset_window[0, :-5] = dataset_window[0, 5:]
	dataset_window[0,-5] = first_action
	dataset_window[0,-4] = -cost
	dataset_window[0, -(observation.shape[0]+1):-1] = observation.T[0]
	dataset_window[0, -1] = mass
	
	u = -1. * np.dot(K, observation)

	step_limit = 0
	while (step_limit < MAXIMUM_NUMBER_OF_STEPS):
		step_limit += 1	
		if trajectories is None:
			trajectories = copy.deepcopy(dataset_window)
			expert_actions = u
		else:
			trajectories = np.append(trajectories, dataset_window, axis=0)
			expert_actions = np.append(expert_actions, u, axis=0)

		observation, cost, finish = env.step(u)

		if not window_size == 1:
			dataset_window[0, :-5] = dataset_window[0, 5:]
			dataset_window[0,-5] = u
			dataset_window[0,-4] = -cost
		dataset_window[0, -(observation.shape[0]+1):-1] = observation.T[0]
		dataset_window[0, -1] = mass
		
		u = -1. * np.dot(K, observation)
	
	return trajectories, expert_actions


def getExpertDatasetFromMass(mass, window_size):
	## Initial-States Grid
	all_states, all_velocities = np.meshgrid(np.linspace(-5, 5, 11), np.linspace(-5, 5, 11))
	all_states = np.reshape(all_states, (-1, 1))
	all_velocities = np.reshape(all_velocities, (-1, 1))
	all_initial_states = np.append(all_states, all_velocities, axis = 1)

	all_trajectories_for_given_mass = None
	all_expert_actions_for_given_mass = None

	for initial_state in all_initial_states:
		trajectories, expert_actions = getExpertDatasetFromMassNInitialState(mass=mass, initial_state=initial_state, window_size= window_size)
		
		if all_trajectories_for_given_mass is None:
			all_trajectories_for_given_mass = copy.deepcopy(trajectories)
			all_expert_actions_for_given_mass = copy.deepcopy(expert_actions)
		else:
			all_trajectories_for_given_mass = np.append(all_trajectories_for_given_mass, trajectories, axis=0)
			all_expert_actions_for_given_mass = np.append(all_expert_actions_for_given_mass, expert_actions, axis=0)

	return all_trajectories_for_given_mass, all_expert_actions_for_given_mass


def getExpertDataset(all_block_masses, window_size):
	
	all_trajectories = None
	all_expert_actions = None

	for block_mass in all_block_masses:
		trajectories, expert_actions = getExpertDatasetFromMass(mass=block_mass, window_size=window_size)
		if all_trajectories is None:
			all_trajectories = copy.deepcopy(trajectories)
			all_expert_actions = copy.deepcopy(expert_actions)
		else:
			all_trajectories = np.append(all_trajectories, trajectories, axis=0)
			all_expert_actions = np.append(all_expert_actions, expert_actions, axis=0)

	return all_trajectories, all_expert_actions, 5