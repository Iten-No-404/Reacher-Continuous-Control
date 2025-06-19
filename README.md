# Reacher-Continuous-Control
This repository is my implementation for the second project of Udacity's Deep Reinforcement Learning Nanodegree. The GIF below shows the performance of the best model which achieved a test mean score of `34.94` and the environment was solved at episode **398**. More details regarding the architecture of the model, its best hyperparameters, results, and future work can be found in [this report](./Report.pdf).
![Trained Agent](./media/best_model.gif)

## 💡 Project Details
This project uses version 2 of Unity's ML Agents Reacher environment which contains 20 identical agents, each with its own copy of the environment. The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

To solve the second version of the environment, we need to take into account the presence of many agents.  In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

## 👩🏻‍💻 Getting Started
To set up your python environment to run the code in this repository, follow the instructions below. 

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	conda activate drlnd
	```
	
2. If running in **Windows**, ensure you have the "Build Tools for Visual Studio 2019" installed from this [site](https://visualstudio.microsoft.com/downloads/).  This [article](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) may also be very helpful.  This was confirmed to work in Windows 10 Home.  

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
4. Clone the DRL repository, and navigate to the `python/` folder.  Then, install several dependencies.  
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```
5. Clone this repository to be able to run its code and use its models.
    ```bash
    git clone https://github.com/Iten-No-404/Reacher-Continuous-Control.git
    ```

Note that this installation guildline is adapted from [Udacity's DRL GitHub repository](https://github.com/udacity/deep-reinforcement-learning)

## 🧭 How to use?

Disclaimer, some of this code is adapted from the official [Udacity's DRL GitHub repository](https://github.com/udacity/deep-reinforcement-learning).

The repository is structured as follows:
- [Continuous_Control.ipynb](./Continuous_Control.ipynb) contains an introduction to the Reacher Navigation environment and how to use it with Unity's ML Agents.
- [model.py](./model.py) defines the Actor and Critic network architecture.
- [ddpg_agent.py](./ddpg_agent.py) defines the DDPG agent that can be used for version 1 of the Reacher environment as well as the ReplayBuffer class for handling the experience replay buffer logic. It also has a class for the Ornstein Uhlenbeck Action Noise process (OUNoise) needed for a more realistic portrayal of the actions. 
- [ddpg_agent_multi.py](./ddpg_agent_multi.py) defines the D4PG agent class. It contains a single Critic Network and one Actor Network but contains an extra multiple learning functionality. Both networks share the same is experience replay buffer.
- [utils.py](./utils.py) contains a small helping function for creating ordered directories.
- [DRL.ipynb](./DRL.ipynb) contains the training loop and the hyperparameter attempts on version 1 of the Reacher environment.
- [DRL_Multi.ipynb](./DRL_Multi.ipynb) contains the training loop and the hyperparameter attempts on version 2 of the Reacher environment.
- [Test_Model.ipynb](./Test_Model.ipynb) contains a simple loop to test the trained models and measure their mean scores over 100 episodes as well as create visualizations.
- [Report.pdf](./Report.pdf) describes the used learning algorithm, the best hyperparameters, results and future work.
- [The ddpg_trials](.ddpg_trials) directory includes the results of some of the 18 training trials as some were force terminated early due to low performance. Each subfolder indicates a trial and contain the following:
  - [critic.pth](./ddpg_trials/18/critic.pth) which are the saved weights of the Critic model after training.
  - [actor.pth](./ddpg_trials/18/0_actor.pth) which are the saved weights of the Actor model after training.
  - [parameters.json](./ddpg_trials/18/parameters.json) which contains the hyperparameters used in training this model.
  - [scores.json](./ddpg_trials/18/scores.json) contains all the scores (i.e. rewards) achieved in each training episode.
  - [test_scores.json](./ddpg_trials/18/test_scores.json) contains all the scores (i.e. rewards) achieved in each testing episode (from the 100 test episodes) of each agent and the mean scores over all agents.
  
Note that the above links are for the best model whose subfolder is `18`.
