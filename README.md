# Uncertainty Aware Imitation Learning on Multiple Tasks using Bayesian Neural Networks

## MuJoCo Experiments

##### Set-up
1. The experiments have been performed on the following packages:
	- openai gym, version = '0.9.3'
	- MuJoCo mjpro131

2. Some experiments need changes to be made to the openai gym source code. Perform one of the following operations to have things set up for yourself.
	- Copy, paste (, and replace if asked) each and every file inside `MuJoCo/gym_files_to_be_merged` to their equivalent locations of your gym installation.
	- Use the OpenAI gym source code from [here](https://github.com/sanjaythakur/Multiple_Task_MuJoCo_Domains).
	Note that the new files needed for the experiment does not break any other part of the original code.

##### Setting up the demonstrator controllers
1. There are 20 tasks defined both for Swimmer and HalfCheetah domains. In order to generate demonstrators on these tasks use the script `train.py` under `MuJoCo/PPO` or just run `./overnight_1.sh` for generating HalfCheetah demonstrators and `./overnight_2.sh` for generating Swimmer demonstrators. The generated demonstrator controllers are stored under `MuJoCo/saved_demonstrator_models` directory. Note that at the end of training a new demonstrator controller, a certain number of demonstration episodes are run and stored for later use under the directory `MuJoCo/demonstrator_trajectories`. The number of demonstration episodes to record can be changed by changing the value of variable `DEMONSTRATOR_EPISODES_TO_LOG` in the file `Housekeeping.py`.
Note that the code for PPO has been taken from [here](https://github.com/sanjaythakur/trpo).

2. The quality of demonstrators can be checked by running the script `test_demonstrator_models.py` under `MuJoCo/PPO`. This script will generate plots showing performance of controllers under the directory `MuJoCo/demonstrator_controller_reward_log`. An example usage is ```python test_demonstrator_models.py HalfCheetah```.

##### Running the proposed mechanism and the naive controller
1. The script for running the proposed mechanism and naive controller is `MuJoCo/data_efficiency_across_multiple_tasks.py` where the domain can be specified using the argument `-dn`. An example usage would look like this ```python data_efficiency_across_multiple_tasks.py -dn Swimmer```. Other important arguments are 
	- `-adt` --> True/False --> Whether or not to make the detector adaptive,
	- `-dc` --> Any float value --> Scaling factor for the detector threshold,
	- `-dm` --> Any integer value --> Number of last few time-steps to smoothen predictive uncertainty,
	- `-idt` --> Any float value --> The detector threshold to start with or the value of non-adaptive threshold.
	The simulation is run over *TOTAL_SIMULATION_ITERATIONS* number of times whose value can be set in the file `Housekeeping.py`. All the pertinent information like rewards obtained, state space traversed etc, the imitation controllers are logged under the directory `MuJoCo/logs` in a systematic way.

##### Comparing the learnability and generalizability of Gaussian Processes (GPs) and Bayes-By-Backprops (BBBs) on MuJoCo




## Sliding Block Experiments