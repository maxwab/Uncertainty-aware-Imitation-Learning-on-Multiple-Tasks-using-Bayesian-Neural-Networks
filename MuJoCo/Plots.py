import matplotlib.pyplot as plt
import numpy as np, os, sys, _pickle as pickle

import sys
#sys.path.insert(0,'./../Task_Agnostic_Online_Multitask_Imitation_Learning/')
sys.path.insert(0,'./../')
from Housekeeping import *


def compare_data_efficiency(all_configurations, demonstration_request_to_gauge):
    simulation_iterator = 0
    x_axis_ticks = []
    cumulative_rewards_to_plot = []
    for configuration in all_configurations:
        if configuration['controller'] == 'bbb_controller':    
            log_file = LOGS_DIRECTORY + configuration['domain_name'] + '/bbb_controller/detector_c_' + str(configuration['detector_c']) + '_detector_m_' + str(configuration['detector_m']) + '/' + str(simulation_iterator) + '/meta_data.pkl'
            with open(log_file, 'rb') as f:
                logged_evaluation_data = pickle.load(f)
            validation_logs_to_probe = LOGS_DIRECTORY + configuration['domain_name'] + '/bbb_controller/detector_c_' + str(configuration['detector_c']) + '_detector_m_' + str(configuration['detector_m']) + '/' + str(simulation_iterator) + '/' + str(logged_evaluation_data[TRAINING_TASK_ITERATION_KEY][demonstration_request_to_gauge-1]) + '/validation_logs.pkl'
            label = '$c=' + str(configuration['detector_c']) + '$\n$m=' + str(configuration['detector_m']) + '$'
        elif configuration['controller'] == 'naive_controller':    
            validation_logs_to_probe = LOGS_DIRECTORY + configuration['domain_name'] + '/naive_controller/' + str(simulation_iterator) + '/' + str(demonstration_request_to_gauge-1) + '/validation_logs.pkl'
            label = 'Naive'
        x_axis_ticks.append(label)
        cumulative_reward_over_all_tasks = 0.
        with open(validation_logs_to_probe, 'rb') as f:
            logged_evaluation_data = pickle.load(f)
        for validation_task in ALL_MUJOCO_TASK_IDENTITIES:
            episodic_rewards = [logged_evaluation_data[str(validation_task)][str(validation_trial)][BEHAVIORAL_CONTROL_REWARDS_LOG_KEY] for validation_trial in range(NUMBER_VALIDATION_TRIALS)]
            episodic_rewards = np.array(episodic_rewards)
            cumulative_reward_over_all_tasks += np.sum(np.mean(episodic_rewards, axis=0))
        cumulative_rewards_to_plot.append(cumulative_reward_over_all_tasks)
    plt.bar(x_axis_ticks, cumulative_rewards_to_plot)
    plt.xlabel('Configuration', fontweight='bold')
    plt.ylabel('$\sum_d^D$', fontweight='bold')
    plt.title('Data-Efficiency', fontsize=18)
    plt.legend()
    plt.show()


def compare_conservativeness(domain_name):
    simulation_iterator = 0

    #log_file = LOGS_DIRECTORY + domain_name + '/naive_controller/' + str(simulation_iterator) + '/meta_data.pkl'
    #with open(log_file, 'rb') as f:
    #    logged_evaluation_data = pickle.load(f)
    #last_trained_task = logged_evaluation_data[TRAINING_TASK_ITERATION_KEY] 

    log_file = LOGS_DIRECTORY + domain_name + '/naive_controller/' + str(simulation_iterator) + '/' + str(len(ALL_MUJOCO_TASK_IDENTITIES)-1) + '/validation_logs.pkl'
    cumulative_reward_over_all_tasks = 0.
    with open(log_file, 'rb') as f:
        logged_evaluation_data = pickle.load(f)

    for validation_task in ALL_MUJOCO_TASK_IDENTITIES:
        episodic_rewards = [logged_evaluation_data[str(validation_task)][str(validation_trial)][BEHAVIORAL_CONTROL_REWARDS_LOG_KEY] for validation_trial in range(NUMBER_VALIDATION_TRIALS)]
        episodic_rewards = np.array(episodic_rewards)
        cumulative_reward_over_all_tasks += np.sum(np.mean(episodic_rewards, axis=0))
        break


def plotDemonstrators(env_name, demonstrator_rewards_over_all_contexts, identifier):
    if env_name == 'Reacher':
        threshold = -3.75
        minimim = -50.
        maximum  = 0.
        random_reward = -44.39
    elif env_name == 'InvertedPendulum':
        threshold = 950.
        minimim = 0.
        maximum = 1100.
        random_reward = 5.2
    elif env_name == 'InvertedDoublePendulum':
        threshold = 9100.
        minimim = 0.
        maximum = 10000.
        random_reward = 53.94
    elif env_name == 'Swimmer':
        threshold = 360.
        minimim = 0.
        maximum = 400.
        random_reward = 1.83
    elif env_name == 'HalfCheetah':
        threshold = 4800.
        minimim = -1000.
        maximum = 7000.
        random_reward = -288.
    elif env_name == 'Hopper':
        threshold = 3800.
        minimim = 0.
        maximum = 4100.
        random_reward = 17.84
    elif env_name == 'Walker2d':
        threshold = 0.
        minimim = 0.
        maximum = 10000.
        random_reward = 1.282601062
    elif env_name == 'Ant':
        threshold = 6000.
        minimim = 0.
        maximum = 8000.
        random_reward = 0.
    elif env_name == 'Humanoid':
        threshold = 0.
        minimim = 0.
        maximum = 8000.
        random_reward = 116.38
    elif env_name == 'HumanoidStandup':
        threshold = 0.
        minimim = 0.
        maximum = 100000.
        random_reward = 33902.78

    demonstrator_plot_directory = './../' + DEMONSTRATOR_CONTROLLER_REWARD_LOG_DIRECTORY + env_name + '/'
    if not os.path.exists(demonstrator_plot_directory):
        os.makedirs(demonstrator_plot_directory)
    demonstrator_file_name = demonstrator_plot_directory + str(identifier) + '.png'

    plt.plot(ALL_MUJOCO_TASK_IDENTITIES, demonstrator_rewards_over_all_contexts, label='Demonstrator')
    #plt.plot([ALL_MUJOCO_TASK_IDENTITIES[demonstrator_context]], [demonstrator_rewards_over_all_contexts[demonstrator_context]], 'ko', label='demonstrator context')
    plt.axhline(y=threshold, color='r', linestyle='-', label='Threshold for success')
    plt.axhline(y=random_reward, color='g', linestyle='-', label='Random Controller')
    plt.ylim(ymin=minimim)
    plt.ylim(ymax=maximum)
    plt.xlabel('Tasks')
    plt.ylabel('Rewards')
    plt.title('Trained context is ' + str(identifier))
    plt.legend()

    plt.savefig(demonstrator_file_name)
    plt.close('all')


def compare_learnability_in_controllers(all_task_configurations, uncertainty_ylim, cost_ylim, predictive_error_ylim):    
    x_bar_ticks = np.arange(0., ALL_MUJOCO_TASK_IDENTITIES.shape[0])
    fig = plt.figure()
    ax_1 = fig.add_subplot(221, frameon=False)
    ax_2 = fig.add_subplot(222, frameon=False)
    ax_3 = fig.add_subplot(223, frameon=False) 
    for iterator, configuration in enumerate(all_task_configurations):
        if configuration[BEHAVIORAL_CONTROLLER_KEY] == 'RANDOM':
            episodic_rewards_across_tasks = []
            for task_identity in ALL_MUJOCO_TASK_IDENTITIES: 
                controller_log_file = LOGS_DIRECTORY + configuration[DOMAIN_KEY] + '_' + configuration[BEHAVIORAL_CONTROLLER_KEY] + '.pkl'
                with open(controller_log_file, 'rb') as f:
                    controller_log_data = pickle.load(f)
                episodic_rewards_across_tasks = np.mean(controller_log_data, axis=1)
            ax_1.bar(x_bar_ticks, episodic_rewards_across_tasks, label=configuration[BEHAVIORAL_CONTROLLER_KEY], color=matplotlibcolors[iterator], width=barwidth/1.25, edgecolor='white')        
        else:
            episodic_rewards_across_tasks, episodic_deviations_across_tasks, episodic_predictive_errors_across_tasks = [], [], []
            for task_identity in ALL_MUJOCO_TASK_IDENTITIES: 
                controller_log_file = LOGS_DIRECTORY + configuration[DOMAIN_KEY] + '_' + str(task_identity) + '_' + configuration[WINDOW_SIZE_KEY] + '_' + configuration[BEHAVIORAL_CONTROLLER_KEY] + '.pkl'
                try:
                    with open(controller_log_file, 'rb') as f:
                        controller_log_data = pickle.load(f)

                    rewards_across_multiple_runs, deviations_across_multiple_runs, episodic_predictive_errors_across_multiple_runs = [], [], []
                    for validation_trial in range(NUMBER_VALIDATION_TRIALS):
                        rewards_across_multiple_runs.append(controller_log_data[str(task_identity)][str(validation_trial)][BEHAVIORAL_CONTROL_REWARDS_LOG_KEY])
                        deviations_across_multiple_runs.append(np.mean(controller_log_data[str(task_identity)][str(validation_trial)][BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY], axis=1))
                        episodic_predictive_errors_across_multiple_runs.append(np.mean(np.square(controller_log_data[str(task_identity)][str(validation_trial)][BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY] - controller_log_data[str(task_identity)][str(validation_trial)][TARGET_CONTROL_MEANS_LOG_KEY]), axis=1))

                    episodic_rewards_across_tasks.append(np.sum(np.mean(rewards_across_multiple_runs, axis=0)))
                    episodic_deviations_across_tasks.append(np.sum(np.mean(deviations_across_multiple_runs, axis=0)))
                    episodic_predictive_errors_across_tasks.append(np.sum(np.mean(episodic_predictive_errors_across_multiple_runs, axis=0)))
                except IOError:
                    episodic_rewards_across_tasks.append(0.)
                    episodic_deviations_across_tasks.append(0.)
                    episodic_predictive_errors_across_tasks.append(0.)

            episodic_rewards_across_tasks = np.array(episodic_rewards_across_tasks)
            episodic_deviations_across_tasks = np.array(episodic_deviations_across_tasks)
            episodic_predictive_errors_across_tasks = np.array(episodic_predictive_errors_across_tasks)
            ax_1.bar(x_bar_ticks, episodic_rewards_across_tasks, label=configuration[BEHAVIORAL_CONTROLLER_KEY], color=matplotlibcolors[iterator], width=barwidth/1.25, edgecolor='white')
            ax_2.bar(x_bar_ticks, episodic_deviations_across_tasks, label=configuration[BEHAVIORAL_CONTROLLER_KEY], color=matplotlibcolors[iterator], width=barwidth/1.25, edgecolor='white')
            ax_3.bar(x_bar_ticks, episodic_predictive_errors_across_tasks, label=configuration[BEHAVIORAL_CONTROLLER_KEY], color=matplotlibcolors[iterator], width=barwidth/1.25, edgecolor='white')
        
        x_bar_ticks = [x + barwidth for x in x_bar_ticks]
        
    ax_1.set_xlabel('Tasks', fontweight='bold')
    ax_1.set_xticks(np.linspace(0., episodic_rewards_across_tasks.shape[0]-1, episodic_rewards_across_tasks.shape[0]))
    ax_1.set_ylabel('Episodic Rewards', fontweight='bold')
    #ax_1.set_yticks(np.arange(0., 3.0, 0.2))
    #ax_1.set_yscale('log')
    #ax_1.set_ylim(0., predictive_error_ylim)
    #ax_1.set_title('Episodic Rewards')
    ax_1.legend()
    
    ax_2.set_xlabel('Tasks', fontweight='bold')
    ax_2.set_xticks(np.linspace(0., episodic_deviations_across_tasks.shape[0]-1, episodic_deviations_across_tasks.shape[0]))
    ax_2.set_ylabel('Episodic Standard Deviation', fontweight='bold')
    #ax_2.set_yscale('log')
    #ax_2.set_title('Episodic Standard Deviation')
    ax_2.legend()
    
    ax_3.set_xlabel('Tasks', fontweight='bold')
    ax_3.set_xticks(np.linspace(0., episodic_predictive_errors_across_tasks.shape[0]-1, episodic_predictive_errors_across_tasks.shape[0]))
    ax_3.set_ylabel('Episodic Mean Squared Predictive Errors', fontweight='bold')
    #ax_3.set_yscale('log')
    #ax_3.set_title('Episodic Standard Deviation')
    ax_3.legend()
    
    fig.suptitle('Relative Learnability of BBB and GP on ' + configuration[DOMAIN_KEY] + ' domain', fontsize=40)
    
    plt.show()    



def compare_generalization_in_controllers(all_task_configurations, uncertainty_ylim, cost_ylim, predictive_error_ylim):    
    gp_exists = False
    fig = plt.figure()
    ax_1 = fig.add_subplot(321, frameon=False)
    ax_2 = fig.add_subplot(322, frameon=False)
    ax_3 = fig.add_subplot(323, frameon=False) 
    for iterator, configuration in enumerate(all_task_configurations):
        if configuration[BEHAVIORAL_CONTROLLER_KEY] == 'RANDOM':
            controller_log_file = LOGS_DIRECTORY + configuration[DOMAIN_KEY] + '_' + configuration[BEHAVIORAL_CONTROLLER_KEY] + '.pkl'
            with open(controller_log_file, 'rb') as f:
                controller_log_data = pickle.load(f)
            episodic_rewards_across_tasks = np.mean(controller_log_data, axis=1)
            ax_1.plot(np.arange(0., episodic_rewards_across_tasks.shape[0]), episodic_rewards_across_tasks, label=configuration[BEHAVIORAL_CONTROLLER_KEY] + ' controller', color=matplotlibcolors[iterator])
        else:   
            controller_log_file = LOGS_DIRECTORY + configuration[DOMAIN_KEY] + '_' + configuration[BEHAVIORAL_CONTROLLER_PROFILE_KEY] + '_' + configuration[WINDOW_SIZE_KEY] + '_' + configuration[BEHAVIORAL_CONTROLLER_KEY] + '.pkl'
            with open(controller_log_file, 'rb') as f:
                controller_log_data = pickle.load(f)
            episodic_rewards_across_tasks, episodic_deviations_across_tasks, episodic_predictive_errors_across_tasks = [], [], []
            for task_to_validate in ALL_MUJOCO_TASK_IDENTITIES:
                rewards_across_multiple_runs, deviations_across_multiple_runs, episodic_predictive_errors_across_multiple_runs = [], [], []
                for validation_trial in range(NUMBER_VALIDATION_TRIALS):
                    rewards_across_multiple_runs.append(controller_log_data[str(task_to_validate)][str(validation_trial)][BEHAVIORAL_CONTROL_REWARDS_LOG_KEY])
                    deviations_across_multiple_runs.append(np.mean(controller_log_data[str(task_to_validate)][str(validation_trial)][BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY], axis=1))
                    episodic_predictive_errors_across_multiple_runs.append(np.mean(np.square(controller_log_data[str(task_to_validate)][str(validation_trial)][BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY] - controller_log_data[str(task_to_validate)][str(validation_trial)][TARGET_CONTROL_MEANS_LOG_KEY]), axis=1))
                episodic_rewards_across_tasks.append(np.sum(np.mean(rewards_across_multiple_runs, axis=0)))
                episodic_deviations_across_tasks.append(np.sum(np.mean(deviations_across_multiple_runs, axis=0)))
                episodic_predictive_errors_across_tasks.append(np.sum(np.mean(episodic_predictive_errors_across_multiple_runs, axis=0)))
            episodic_rewards_across_tasks = np.array(episodic_rewards_across_tasks)
            episodic_deviations_across_tasks = np.array(episodic_deviations_across_tasks)
            episodic_predictive_errors_across_tasks = np.array(episodic_predictive_errors_across_tasks)
            ax_1.plot(np.arange(0., episodic_rewards_across_tasks.shape[0]), episodic_rewards_across_tasks, label=configuration[BEHAVIORAL_CONTROLLER_KEY] + ' controller, trained on task ' + str(configuration[BEHAVIORAL_CONTROLLER_PROFILE_KEY]), color=matplotlibcolors[iterator])
            ax_2.plot(np.arange(0., episodic_deviations_across_tasks.shape[0]), episodic_deviations_across_tasks, label=configuration[BEHAVIORAL_CONTROLLER_KEY] + ' controller, trained on task ' + str(configuration[BEHAVIORAL_CONTROLLER_PROFILE_KEY]), color=matplotlibcolors[iterator])
            ax_3.plot(np.arange(0., episodic_predictive_errors_across_tasks.shape[0]), episodic_predictive_errors_across_tasks, label=configuration[BEHAVIORAL_CONTROLLER_KEY] + ' controller, trained on task ' + str(configuration[BEHAVIORAL_CONTROLLER_PROFILE_KEY]), color=matplotlibcolors[iterator])
    
        if configuration[BEHAVIORAL_CONTROLLER_KEY] == 'GP':
            if not gp_exists:
                gp_exists = True
                ax_4 = fig.add_subplot(324, frameon=False)
                ax_5 = fig.add_subplot(325, frameon=False)
            gp_fit_logs_file = LOGS_DIRECTORY + configuration[DOMAIN_KEY] + '_' + configuration[BEHAVIORAL_CONTROLLER_PROFILE_KEY] + '_' + configuration[WINDOW_SIZE_KEY] + '_GP_fit.pkl'
            with open(gp_fit_logs_file, 'rb') as f:
                gp_fit_logs_data = pickle.load(f)
            unoptimized_gp_fit_data = gp_fit_logs_data[UNOPTIMIZED_GP_FIT_KEY]
            unoptimized_gp_trainables_data  = gp_fit_logs_data[UNOPTIMIZED_GP_TRAINABLES_KEY]
            optimized_gp_fit_data = gp_fit_logs_data[OPTIMIZED_GP_FIT_KEY]
            optimized_gp_trainables_data  = gp_fit_logs_data[OPTIMIZED_GP_TRAINABLES_KEY]
            print(RED('Trainables before optimization'))
            print(unoptimized_gp_trainables_data)
            print(RED('Trainables after maximization of marginalized likelihood by marginalizing GP hyperparameters'))
            print(optimized_gp_trainables_data)
            mean_gp_fit_predictive_error = [unoptimized_gp_fit_data[MEAN_GP_FIT_PREDICTIVE_ERROR_KEY], optimized_gp_fit_data[MEAN_GP_FIT_PREDICTIVE_ERROR_KEY]]
            mean_gp_fit_predictive_variance = [unoptimized_gp_fit_data[MEAN_GP_FIT_VARIANCE_KEY], optimized_gp_fit_data[MEAN_GP_FIT_VARIANCE_KEY]]
            
            x_bar_labels = ['Unoptimized', 'Optimized']
            ax_4.bar(x_bar_labels, mean_gp_fit_predictive_error, width=barwidth/1.25, edgecolor='white')
            ax_5.bar(x_bar_labels, mean_gp_fit_predictive_variance, width=barwidth/1.25, edgecolor='white')
    
    ax_1.set_xlabel('Task Identifiers', fontweight='bold')
    ax_1.set_xticks(np.linspace(0., episodic_rewards_across_tasks.shape[0]-1, episodic_rewards_across_tasks.shape[0]))
    ax_1.set_ylabel('Episodic Rewards', fontweight='bold')
    #ax_1.set_yticks(np.arange(0., 3.0, 0.2))
    #ax_1.set_yscale('log')
    #ax_1.set_ylim(0., predictive_error_ylim)
    ax_1.set_title('Swimmer')
    ax_1.legend()
    
    ax_2.set_xlabel('Task Identifiers', fontweight='bold')
    ax_2.set_xticks(np.linspace(0., episodic_deviations_across_tasks.shape[0]-1, episodic_deviations_across_tasks.shape[0]))
    ax_2.set_ylabel('Episodic Standard Deviation', fontweight='bold')
    #ax_2.set_yscale('log')
    #ax_2.set_title('Episodic Standard Deviation')
    ax_2.legend()
    
    ax_3.set_xlabel('Task Identifiers', fontweight='bold')
    ax_3.set_xticks(np.linspace(0., episodic_predictive_errors_across_tasks.shape[0]-1, episodic_predictive_errors_across_tasks.shape[0]))
    ax_3.set_ylabel('Episodic Mean Squared Predictive Errors', fontweight='bold')
    #ax_3.set_yscale('log')
    #ax_3.set_title('Episodic Standard Deviation')
    ax_3.legend()
    
    #ax_4.set_xlabel('group', fontweight='bold')
    #ax_4.set_xticks(x_bar_ticks_1, x_bar_labels)
    ax_4.set_ylabel('Predictive Mean Squared Error', fontweight='bold')
    ax_4.set_yscale('log')
    ax_4.set_title('GP Optimization Effect on Predictive Error')
    ax_4.legend()

    #ax_5.set_xlabel('group', fontweight='bold')
    #ax_5.set_xticks(x_bar_ticks_1, x_bar_labels)
    ax_5.set_ylabel('Predictive Mean Variance', fontweight='bold')
    ax_5.set_yscale('log')
    ax_5.set_title('GP Optimization Effect on Predictive Variance')
    ax_5.legend()
    
    fig.suptitle('Relative Generalizability of BBB and GP on ' + configuration[DOMAIN_KEY] + ' domain', fontsize=40)
    
    plt.show()    


def inspect_a_learner(configuration, uncertainty_ylim, cost_ylim, predictive_error_ylim):    
    fig = plt.figure()
    ax_1 = fig.add_subplot(221, frameon=False)
    ax_2 = fig.add_subplot(222, frameon=False)  
    
    controller_log_file = LOGS_DIRECTORY + configuration[DOMAIN_KEY] + '_' + configuration[BEHAVIORAL_CONTROLLER_PROFILE_KEY] + '_' + configuration[WINDOW_SIZE_KEY] + '_' + configuration[BEHAVIORAL_CONTROLLER_KEY] + '.pkl'
    with open(controller_log_file, 'rb') as f:
        controller_log_data = pickle.load(f)
        episodic_rewards_across_tasks, episodic_deviations_across_tasks = [], []
        for task_to_validate in ALL_MUJOCO_CONTEXTS:
            rewards_across_multiple_runs, deviations_across_multiple_runs = [], []
            for validation_trial in range(NUMBER_VALIDATION_TRIALS):
                rewards_across_multiple_runs.append(controller_log_data[str(task_to_validate)][str(validation_trial)][BEHAVIORAL_CONTROL_REWARDS_LOG_KEY])
                deviations_across_multiple_runs.append(np.mean(controller_log_data[str(task_to_validate)][str(validation_trial)][BEHAVIORAL_CONTROL_DEVIATIONS_LOG_KEY], axis=1))
            per_time_step_rewards_across_tasks = np.mean(rewards_across_multiple_runs, axis=0)
            per_time_step_deviations_across_tasks = np.mean(deviations_across_multiple_runs, axis=0)
            ax_1.plot(np.arange(0., per_time_step_rewards_across_tasks.shape[0]), per_time_step_rewards_across_tasks, label='Task Identity: ' + str(task_to_validate))
            ax_2.plot(np.arange(0., per_time_step_deviations_across_tasks.shape[0]), per_time_step_deviations_across_tasks, label='Task Identity: ' + str(task_to_validate))

    ax_1.set_xlabel('Time-Steps', fontweight='bold')
    ax_1.set_xticks(np.linspace(0., per_time_step_rewards_across_tasks.shape[0], 11.))
    ax_1.set_ylabel('Rewards', fontweight='bold')
    #ax_1.set_yticks(np.arange(0., 3.0, 0.2))
    #ax_1.set_yscale('log')
    #ax_1.set_ylim(0., uncertainty_ylim)
    ax_1.set_title('Per time-step rewards')
    ax_1.legend()
    ax_2.set_xlabel('Time-Steps', fontweight='bold')
    ax_2.set_xticks(np.linspace(0., per_time_step_deviations_across_tasks.shape[0], 11.))
    ax_2.set_ylabel('Standard Deviation', fontweight='bold')
    #ax_2.set_yticks(np.arange(0., 3.0, 0.2))
    #ax_2.set_yscale('log')
    #ax_2.set_ylim(0., cost_ylim)
    ax_2.set_title('Per time-step deviations')
    ax_2.legend()
    
    plt.show()    