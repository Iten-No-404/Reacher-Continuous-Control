import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from ddpg_agent import ReplayBuffer, OUNoise, device

import torch
import torch.nn.functional as F
import torch.optim as optim

# Assuming Actor, Critic, OUNoise, and device are defined elsewhere

class SharedAgent:
    def __init__(self, state_size, action_size, num_agents, random_seed=0,
                 actor_lr=1e-4, critic_lr=1e-3, weight_decay=0.0,
                 buffer_size=int(1e6), batch_size=128, gamma=0.99, tau=1e-3,
                 learn_every=20, learn_iterations=10):

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learn_every = learn_every
        self.learn_iterations = learn_iterations
        self.t_step = 0  # time step counter

        # Actor Network (shared)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)

        # Critic Network (shared)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr, weight_decay=weight_decay)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)

        # Shared noise process
        self.noise = OUNoise((num_agents, action_size), random_seed)

    def act(self, states, add_noise=True):
        """Returns actions for all agents."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experiences in replay buffer and trigger learning."""
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        self.t_step += 1

        # Learn every `learn_every` steps
        if self.t_step % self.learn_every == 0 and len(self.memory) > self.batch_size:
            for _ in range(self.learn_iterations):
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        """Update policy and value parameters using batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def reset(self):
        self.noise.reset()
