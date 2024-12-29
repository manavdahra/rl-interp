import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from .dqn import DQN
from .replay_buffer import ReplayBuffer

class Agent:
    def __init__(
        self, 
        state_size, 
        action_size, 
        batch_size=128,
        gamma=0.99, # discount factor
        epsilon=1.0, # exploration rate
        epsilon_min=0.05,
        epsilon_decay=0.995,
        tau=0.01,
        learning_rate=1e-4,
        replay_buffer=10000,
        restore=False, 
        data_dir="data_dir"
    ):
        print(f"Initialising agent with State size: {state_size} Action size: {action_size}")
        
        self.state_size = state_size
        self.action_size = action_size
        self.data_dir = data_dir
        
        # Hyperparameters
        self.gamma = gamma  
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory = ReplayBuffer(replay_buffer)
        
        # Networks
        self.policy_net = DQN("policy_net", state_size, action_size, self.data_dir)
        self.target_net = DQN("target_net", state_size, action_size, self.data_dir)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        if restore:
            self.load_agent()    
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        self._init_data_dir()
    
    def _init_data_dir(self):
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _select_best_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
        
    def select_action(self, state, is_eval=False):
        if is_eval:
            return self._select_best_action(state)
        
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        return self._select_best_action(state)
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        self.policy_net.train()
        self.target_net.train()
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        policy_net_state = self.policy_net.state_dict()
        target_net_state = self.target_net.state_dict()
        for key in policy_net_state:
            target_net_state[key] = policy_net_state[key]*self.tau + target_net_state[key]*(1.0 - self.tau)
        self.target_net.load_state_dict(target_net_state)
        
    def save_agent(self):
        print(f"Saving the agent state")
        self.policy_net.save_model()
        self.target_net.save_model()
    
    def load_agent(self):
        print(f"Loading agent state")
        self.policy_net.load_model()
        self.target_net.load_model()
