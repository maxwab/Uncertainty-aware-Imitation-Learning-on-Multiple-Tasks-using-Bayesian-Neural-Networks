import gym, numpy as np, argparse


def get_contextual_HalfCheetah_by_mass(context):
	env = gym.make('HalfCheetah-v1')
	body_mass = env.env.model.body_mass
	body_mass = np.array(body_mass)

	if context == '0':
		# Keeping the default settings
		pass
	elif context == '1':
		body_mass[1,0] = 10.
		body_mass[2,0] = 5.
	elif context == '2':
		body_mass[1,0] = 0.5
	elif context == '3':
		body_mass[2,0] = 4.5
	elif context == '4':
		body_mass[2,0] = 3.
	elif context == '5':
		body_mass[1,0] = 0.8
	elif context == '6':
		body_mass[1,0] = 0.8
		body_mass[2,0] = 0.5
	elif context == '7':
		body_mass[1,0] = 0.5
		body_mass[2,0] = 0.5
	elif context == '8':
		body_mass[1,0] = 10.
		body_mass[2,0] = 4.5
	elif context == '9':
		body_mass[1,0] = 10.
		body_mass[2,0] = 6.
	else:
		print('HalfCheetah-v1 not set for context ' + context + '. Program is exiting now...')
		exit(0)

	env.env.model.body_mass = body_mass
	return env



def get_contextual_Swimmer_by_mass(context):
	env = gym.make('Swimmer-v1')
	body_mass = env.env.model.body_mass
	body_mass = np.array(body_mass)

	if context == '0':
		# Keeping the default settings
		pass
	elif context == '1':
		body_mass[0,0] = 10.
	elif context == '2':
		body_mass[1,0] = 10.
	elif context == '3':
		body_mass[2,0] = 10.
	elif context == '4':
		body_mass[3,0] = 10.
	elif context == '5':
		body_mass[2,0] = 28.
	elif context == '6':
		body_mass[2,0] = 32.
	elif context == '7':
		body_mass[3,0] = 32.
	elif context == '8':
		body_mass[2,0] = 25.
	elif context == '9':
		body_mass[1,0] = 30.
	else:
		print('Swimmer-v1 not set for context ' + context + '. Program is exiting now...')
		exit(0)

	env.env.model.body_mass = body_mass
	return env


def get_contextual_Swimmer_by_length(context):
    if context == '0':
        # Keeping the default settings
        env = gym.make('Swimmer-v1')
    elif context == '1':
        env = gym.make('Swimmer_1-v1')
    elif context == '2':
        env = gym.make('Swimmer_2-v1')
    elif context == '3':
        env = gym.make('Swimmer_3-v1')
    elif context == '4':
        env = gym.make('Swimmer_4-v1')
    elif context == '5':
        env = gym.make('Swimmer_5-v1')
    elif context == '6':
        env = gym.make('Swimmer_6-v1')
    elif context == '7':
        env = gym.make('Swimmer_7-v1')
    elif context == '8':
        env = gym.make('Swimmer_8-v1')
    elif context == '9':
        env = gym.make('Swimmer_9-v1')
    else:
        print('Swimmer-v1 not set for context ' + context + '. Program is exiting now...')
        exit(0)

    return env


def get_contextual_Swimmer(context):
	if context == '0' or context == '1' or context == '2' or context == '3' or context == '4':
		env = get_contextual_Swimmer_by_length(context)
	else:
		env = get_contextual_Swimmer_by_mass(context)
	return env


def get_contextual_Walker2d_by_mass(context):
	env = gym.make('Walker2d-v1')
	body_mass = env.env.model.body_mass
	body_mass = np.array(body_mass)

	if context == '0':
		# Keeping the default settings
		pass
	elif context == '1':
		body_mass[1,0] = 35. 
	elif context == '2':
		body_mass[3,0] = 27.
	elif context == '3':
		body_mass[5,0] = 40.
	elif context == '4':
		body_mass[7,0] = 30.
	elif context == '5':
		body_mass[1,0] = 0.35
	elif context == '6':
		body_mass[3,0] = 0.27
	elif context == '7':
		body_mass[5,0] = 0.4
	elif context == '8':
		body_mass[7,0] = 0.3
	elif context == '9':
		body_mass[1,0] = 35.
		body_mass[5,0] = .4
	else:
		print('Walker2d-v1 not set for context ' + context + '. Program is exiting now...')
		exit(0)

	env.env.model.body_mass = body_mass
	return env


def get_contextual_HumanoidStandup_by_mass(context):
	env = gym.make('HumanoidStandup-v1')
	body_mass = env.env.model.body_mass
	body_mass = np.array(body_mass)

	if context == '0':
		# Keeping the default settings
		pass
	elif context == '1':
		body_mass[1,0] = 83.
	elif context == '2':
		body_mass[4,0] = 45.
	elif context == '3':
		body_mass[8,0] = 26.
	elif context == '4':
		body_mass[11,0] = 11.
	elif context == '5':
		body_mass[12,0] = 15.
	elif context == '6':
		body_mass[1,0] = 0.83
	elif context == '7':
		body_mass[4,0] = 0.45
	elif context == '8':
		body_mass[8,0] = 0.26
	elif context == '9':
		body_mass[12,0] = 0.15
	else:
		print('HumanoidStandup-v1 not set for context ' + context + '. Program is exiting now...')
		exit(0)

	env.env.model.body_mass = body_mass
	return env


def get_contextual_Humanoid_by_mass(context):
	env = gym.make('Humanoid-v1')
	body_mass = env.env.model.body_mass
	body_mass = np.array(body_mass)

	if context == '0':
		# Keeping the default settings
		pass
	elif context == '1':
		body_mass[1,0] = 83.
	elif context == '2':
		body_mass[4,0] = 45.
	elif context == '3':
		body_mass[8,0] = 26.
	elif context == '4':
		body_mass[11,0] = 11.
	elif context == '5':
		body_mass[12,0] = 15.
	elif context == '6':
		body_mass[1,0] = 0.83
	elif context == '7':
		body_mass[4,0] = 0.45
	elif context == '8':
		body_mass[8,0] = 0.26
	elif context == '9':
		body_mass[12,0] = 0.15
	else:
		print('Humanoid-v1 not set for context ' + context + '. Program is exiting now...')
		exit(0)

	env.env.model.body_mass = body_mass
	return env


def get_contextual_Ant_by_mass(context):
	env = gym.make('Ant-v1')
	body_mass = env.env.model.body_mass
	body_mass = np.array(body_mass)

	if context == '0':
		# Keeping the default settings
		pass
	elif context == '1':
		body_mass[1,0] = 3.2
	elif context == '2':
		body_mass[4,0] = 0.6
	elif context == '3':
		body_mass[8,0] = 0.3
	elif context == '4':
		body_mass[11,0] = 0.3
	elif context == '5':
		body_mass[12,0] = 0.3
	elif context == '6':
		body_mass[1,0] = 0.03
	elif context == '7':
		body_mass[4,0] = 0.006
	elif context == '8':
		body_mass[8,0] = 0.003
	elif context == '9':
		body_mass[12,0] = 0.003
	else:
		print('Ant-v1 not set for context ' + context + '. Program is exiting now...')
		exit(0)

	env.env.model.body_mass = body_mass
	return env


def get_contextual_Hopper_by_mass(context):
	env = gym.make('Hopper-v1')
	body_mass = env.env.model.body_mass
	body_mass = np.array(body_mass)

	if context == '0':
		# Keeping the default settings
		pass
	elif context == '1':
		body_mass[0,0] = 10.
	elif context == '2':
		body_mass[1,0] = 35.
	elif context == '3':
		body_mass[2,0] = 40.
	elif context == '4':
		body_mass[3,0] = 30.
	elif context == '5':
		body_mass[4,0] = 50.
	elif context == '6':
		body_mass[1,0] = 0.35
	elif context == '7':
		body_mass[2,0] = 0.4
	elif context == '8':
		body_mass[3,0] = 0.3
	elif context == '9':
		body_mass[4,0] = 0.5
	else:
		print('Hopper-v1 not set for context ' + context + '. Program is exiting now...')
		exit(0)

	env.env.model.body_mass = body_mass
	return env


def get_contextual_InvertedDoublePendulum_by_mass(context):
	env = gym.make('InvertedDoublePendulum-v1')
	body_mass = env.env.model.body_mass
	body_mass = np.array(body_mass)

	if context == '0':
		# Keeping the default settings
		pass
	elif context == '1':
		body_mass[0,0] = 10.
	elif context == '2':
		body_mass[1,0] = 100.
	elif context == '3':
		body_mass[2,0] = 40.
	elif context == '4':
		body_mass[3,0] = 40.
	elif context == '5':
		body_mass[1,0] = 1.
	elif context == '6':
		body_mass[2,0] = .6
	elif context == '7':
		body_mass[3,0] = .6
	elif context == '8':
		body_mass[2,0] = 40.
		body_mass[3,0] = 40.
	elif context == '9':
		body_mass[2,0] = 0.6
		body_mass[3,0] = 40.
	else:
		print('InvertedDoublePendulum-v1 not set for context ' + context + '. Program is exiting now...')
		exit(0)

	env.env.model.body_mass = body_mass
	return env


def get_contextual_InvertedPendulum_by_mass(context):
	env = gym.make('InvertedPendulum-v1')
	body_mass = env.env.model.body_mass
	body_mass = np.array(body_mass)

	if context == '0':
		# Keeping the default settings
		pass
	elif context == '1':
		body_mass[0,0] = 10.
	elif context == '2':
		body_mass[1,0] = 100.5
	elif context == '3':
		body_mass[2,0] = 60.
	elif context == '4':
		body_mass[1,0] = .8
	elif context == '5':
		body_mass[2,0] = .3
	elif context == '6':
		body_mass[1,0] = 100.5
		body_mass[2,0] = 60.
	elif context == '7':
		body_mass[1,0] = 100.5
		body_mass[2,0] = .8
	elif context == '8':
		body_mass[1,0] = .8
		body_mass[2,0] = 60.
	elif context == '9':
		body_mass[1,0] = .8
		body_mass[2,0] = .3
	else:
		print('InvertedPendulum-v1 not set for context ' + context + '. Program is exiting now...')
		exit(0)

	env.env.model.body_mass = body_mass
	return env


def get_contextual_Reacher_by_mass(context):
	env = gym.make('Reacher-v1')
	body_mass = env.env.model.body_mass
	body_mass = np.array(body_mass)

	if context == '0':
		# Keeping the default settings
		pass
	elif context == '1':
		body_mass[0,0] = 10.
	elif context == '2':
		body_mass[1,0] = 0.3
	elif context == '3':
		body_mass[2,0] = 0.3
	elif context == '4':
		body_mass[3,0] = 0.04
	elif context == '5':
		body_mass[4,0] = 0.04
	elif context == '6':
		body_mass[1,0] = 0.003
	elif context == '7':
		body_mass[2,0] = 0.003
	elif context == '8':
		body_mass[3,0] = 0.0004
	elif context == '9':
		body_mass[4,0] = 0.0004
	else:
		print('Reacher-v1 not set for context ' + context + '. Program is exiting now...')
		exit(0)

	env.env.model.body_mass = body_mass
	return env


def get_contextual_MUJOCO_environment(env_name, context):
	if env_name == 'Reacher':
		env = get_contextual_Reacher_by_mass(context)
	elif env_name == 'InvertedPendulum':
		env = get_contextual_InvertedPendulum_by_mass(context)
	elif env_name == 'InvertedDoublePendulum':
		env = get_contextual_InvertedDoublePendulum_by_mass(context)
	elif env_name == 'Swimmer':
		env = get_contextual_Swimmer(context)
	elif env_name == 'HalfCheetah':
		env = get_contextual_HalfCheetah_by_mass(context)
	elif env_name == 'Hopper':
		env = get_contextual_Hopper_by_mass(context)
	elif env_name == 'Walker2d':
		env = get_contextual_Walker2d_by_mass(context)
	elif env_name == 'Ant':
		env = get_contextual_Ant_by_mass(context)
	elif env_name == 'Humanoid':
		env = get_contextual_Humanoid_by_mass(context)
	elif env_name == 'HumanoidStandup':
		env = get_contextual_HumanoidStandup_by_mass(context)

	return env


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=('Get modified MUJOCO environment based on your context specified'))
	parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
	parser.add_argument('-c', '--context', type=str,
						help='The underlying and typically unobservable context during operation of a controller',
						default='0')
	args = parser.parse_args()
	
	env = get_contextual_MUJOCO_environment(**vars(args))
	print(env.env.model.body_mass)


#env = gym.make('InvertedPendulum-v1')
#env = gym.make('InvertedDoublePendulum-v1')
#env = gym.make('Reacher-v1')
#env = gym.make('HalfCheetah-v1')
#env = gym.make('Swimmer-v1')
#env = gym.make('Hopper-v1')
#env = gym.make('Walker2d-v1')
#env = gym.make('Ant-v1')
#env = gym.make('Humanoid-v1')
#env = gym.make('HumanoidStandup-v1')

#observation = env.reset()
#print(observation)

#print(env.observation_space)
#print(env.action_space)

#mb = env.env.model.body_mass
#print(mb)


'''

mb = np.array(mb)
mb[3,0] = 70000000.7777
env.model.body_mass = mb
print(env.model.body_mass)

for i_episode in range(10):
	observation = env.reset()
	for t in range(100):
		env.render()
		#print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			#print("Episode finished after {} timesteps".format(t+1))
			break


mb = env.model.body_mass
print(mb)


mb = np.array(mb)
mb[1,0] = 7.7777
env.model.body_mass = mb
print(env.model.body_mass)
'''
