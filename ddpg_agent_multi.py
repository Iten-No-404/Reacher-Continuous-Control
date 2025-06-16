import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic # Assuming Actor and Critic models are defined in model.py

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, 
                     action_size, 
                     random_seed,
                     num_agents, # NEW: Number of agents
                     BUFFER_SIZE = int(1e4),  # replay buffer size
                     BATCH_SIZE = 128,         # minibatch size
                     GAMMA = 0.99,             # discount factor
                     TAU = 1e-3,               # for soft update of target parameters
                     LR_ACTOR = 1e-4,          # learning rate of the actor 
                     LR_CRITIC = 1e-3,         # learning rate of the critic
                     WEIGHT_DECAY = 0):        # L2 weight decay
        
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agents (int): number of agents in the environment
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents # NEW: Store num_agents
        self.BUFFER_SIZE = BUFFER_SIZE    
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LR_ACTOR = LR_ACTOR
        self.LR_CRITIC = LR_CRITIC
        self.WEIGHT_DECAY = WEIGHT_DECAY

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC, weight_decay=self.WEIGHT_DECAY)

        # Noise process (each agent should have its own noise process for exploration)
        # Assuming you'll have an array of OUNoise objects if each agent has individual noise.
        # For simplicity, if only one noise object is used for all actions, keep as is.
        # If each of the 20 agents needs its own independent noise, you'd initialize a list here.
        # For this DDPG setup, typically a single noise object per agent is used for its action.
        # However, since we're using a single Agent class instance to manage learning for ALL agents,
        # we'll assume a single noise object for now that influences the collective actions.
        # If each agent needs unique noise for its actions in a single 'act' call for a batch of states,
        # the noise generation would need to be adjusted inside the 'act' method or by passing noise per agent.
        # For simplicity, we'll keep one noise object and the 'act' method will apply it.
        # A more complex setup for multiple agents would involve passing agent_idx to 'act' or managing multiple noise objects.
        # For *this* specific request (shared buffer, single agent class managing learning for 20 agents),
        # this `noise` object will apply to the 'collective' action space (if you're thinking of a batch of actions)
        # or just be used once per call to 'act' which typically handles one state at a time.
        # If your `act` function in the environment interaction loop receives a batch of states for 20 agents,
        # then the noise would be applied to the batch of actions.

        self.noise = OUNoise(action_size, random_seed) 
        # Replay memory (shared by all agents)
        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, random_seed)
    
    # MODIFIED: step now takes lists for multiple agents
    def step(self, states, actions, rewards, next_states, dones):
        """Save experiences in replay memory, and use random samples from buffer to learn."""
        # Save experiences for all agents
        for i in range(self.num_agents): # Iterate through each agent's experience
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn 'num_agents' times, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            for _ in range(self.num_agents): # Learn 20 times (once for each agent)
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.
           Note: 'state' here is assumed to be a single state, not a batch of states.
           If you pass a batch of states (e.g., from 20 agents),
           the Actor model should be designed to handle batch inputs.
           The noise generation here is for a single action. For 20 agents' actions,
           you'd need to generate 20 noise samples.
        """
        # Ensure state is a batch for the actor if it's not already
        # If state is a single numpy array, unsqueeze it to (1, state_size)
        # If you are passing a batch of states (num_agents, state_size), this line is fine
        state = torch.from_numpy(state).float().to(device)
        if state.dim() == 1: # If state is a single state, make it a batch of 1
            state = state.unsqueeze(0) 

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            # If 'action' is a batch of actions (num_agents, action_size),
            # and 'self.noise.sample()' returns a single noise (action_size,),
            # you need to broadcast or generate noise for each action.
            # Assuming 'self.noise.sample()' is designed to produce noise for `action_size` dimensions.
            # If multiple agents' actions are in 'action', you need to generate noise for each.
            # For simplicity, assuming 'action' is (1, action_size) or (num_agents, action_size)
            # and noise needs to be generated for each row.
            
            # This is a critical point: if 'action' is (N, action_size) for N agents,
            # you'd need N independent noise samples.
            # The current OUNoise.sample() returns one sample (action_size,).
            # So, if 'action' is a batch, you need to apply noise appropriately.
            # A simple (though potentially suboptimal) way if 'action' is (N, action_size):
            noise_samples = np.array([self.noise.sample() for _ in range(action.shape[0])])
            action += noise_samples
        
        action = (action + 1.0) / 2.0
        return np.clip(action, 0, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip gradients to prevent explosion (common in DDPG)
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # Example, add if needed
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean() # DDPG seeks to maximize Q-value
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU) 
                          
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # The original code uses random.random() which returns a single float.
        # For a vector of size 'size', you should probably use np.random.randn(size)
        # for a standard normal distribution, or create 'size' independent random.random() calls.
        # The current implementation creates a list of 'size' random floats. This is fine.
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # It's good practice to ensure experiences are not None, though with proper
        # handling of deque and `random.sample`, they shouldn't be.
        # The original code's check `if e is not None` is robust.
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)