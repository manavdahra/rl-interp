from dataclasses import dataclass
import sys
from typing import List
from cartpole.agent import Agent
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

@dataclass
class State:
    val: List[float]
    
    def value(self):
        return torch.FloatTensor(self.val).unsqueeze(0)
    
    def description(self):
        cols = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]
        index = -1
        value = float("inf")
        for idx, val in enumerate(self.val):
            if val != 0.0:
                index == idx
                value = val
                break
        
        return f"{cols[index]} = {value}"


class RLInterpreter:
    def __init__(self, env_name, seed):
        self.env_name = env_name
        self.seed = seed
        self.env = gym.make(env_name)
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        
        self.agent = Agent(state_size, action_size, batch_size=4, data_dir=f"data_dir/{env_name}")
        self.agent.load_agent()
        
    # Visualizing the learned Q-values for a specific state
    def _visualize_q_values(self, policy_net, state: State):
        q_values = policy_net(state.value()).detach().squeeze()
        max = q_values.max()
        min = q_values.min()
        q_values = (q_values - min) / (max - min)
        actions = ["Left", "Right"]
        
        plt.bar(actions, q_values)
        plt.title(f"Q-values for: {state.description()}")
        plt.ylabel("Q-value")
        plt.show()

    def check_agent_outputs(self):
        # Example state
        # state = State(val=[0.0, 0.0, -0.3, 0.0])
        state = State(val=[1.0, 0.0, 0.0, 0.0])
        self._visualize_q_values(self.agent.policy_net, state)
        
    def create_heatmap_pos_vel(self, bins=20):
        # Discretize the state space (focus on the first two dimensions)
        position = np.linspace(-4.8, 4.8, bins)  # Cart position
        velocity = np.linspace(-4, 4, bins)      # Cart velocity
        q_values_grid = np.zeros((bins, bins))

        for i, pos in enumerate(position):
            for j, vel in enumerate(velocity):
                # Create a state with pos, vel and fixed other dimensions
                state = np.array([pos, vel, 0.0, 0.0])  # Assume other dimensions are zero
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Compute Q-values and take the max
                with torch.no_grad():
                    q_values = self.agent.policy_net(state_tensor)
                
                q_values_grid[i, j] = q_values.argmax().item()
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(q_values_grid, extent=[-4, 4, -4.8, 4.8], origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label='Max Q-value')
        plt.title('Heatmap of Q-values (Position vs Velocity)')
        plt.xlabel('Velocity')
        plt.ylabel('Position')
        plt.show()
        
    def create_heatmap_vel_angle_vel(self, bins=20):
        # Discretize the state space (focus on the first two dimensions)
        velocity = np.linspace(-4, 4, bins)   # Cart velocity
        omega = np.linspace(-4, 4, bins)      # Cart velocity
        q_values_grid = np.zeros((bins, bins))

        for i, v in enumerate(velocity):
            for j, w in enumerate(omega):
                # Create a state with pos, vel and fixed other dimensions
                state = np.array([0.0, v, 0.0, w])  # Assume other dimensions are zero
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Compute Q-values and take the max
                with torch.no_grad():
                    q_values = self.agent.policy_net(state_tensor)
                
                q_values_grid[i, j] = q_values.argmax().item()
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(q_values_grid, extent=[-4, 4, -4, 4], origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label='Max Q-value')
        plt.title('Heatmap of Q-values (Velocity vs Angular Velocity)')
        plt.xlabel('Velocity')
        plt.ylabel('Angular Velocity')
        plt.show()
        
    def create_heatmap_pole_angle_vel(self, bins=20):
        # Discretize the state space (focus on the first two dimensions)
        radians = np.linspace(-0.418, 0.418, bins)  # Cart position
        omega = np.linspace(-4, 4, bins)      # Cart velocity
        q_values_grid = np.zeros((bins, bins))

        for i, r in enumerate(radians):
            for j, w in enumerate(omega):
                # Create a state with pos, vel and fixed other dimensions
                state = np.array([0.0, 0.0, r, w])  # Assume other dimensions are zero
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Compute Q-values and take the max
                with torch.no_grad():
                    q_values = self.agent.policy_net(state_tensor)
                
                q_values_grid[i, j] = q_values.argmax().item()
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(q_values_grid, extent=[-4, 4, -0.418, 0.418], origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label='Max Q-value')
        plt.title('Heatmap of Q-values (Angular Velocity vs Angle)')
        plt.xlabel('Angular Velocity')
        plt.ylabel('Angular Position')
        plt.show()

        
    def interpret(self):
        hidden_1_weights = self.agent.policy_net.state_dict()["network.hidden_1.weight"]
        hidden_2_weights = self.agent.policy_net.state_dict()["network.hidden_2.weight"]
        
        print(hidden_1_weights.shape)
        print(hidden_2_weights.shape)
        
        hidden_1_features = torch.transpose(hidden_1_weights, 0, 1)
        hidden_2_features = torch.transpose(hidden_2_weights, 0, 1)
        
        hidden_1_prod = torch.matmul(hidden_1_features, hidden_1_weights)
        hidden_2_prod = torch.matmul(hidden_2_features, hidden_2_weights)
        
        fig, ax = plt.subplots(1, 2)
        images = []
        images.append(ax[0].imshow(hidden_1_prod, cmap="RdBu"))
        images.append(ax[1].imshow(hidden_2_prod, cmap="RdBu"))
        
        fig.colorbar(images[0], ax=ax, orientation='horizontal')
        plt.show()
    
    def collect_stats(self):
        total_reward = 0
        done = False
        truncated = False
        state, _ = self.env.reset(seed=self.seed)
        while not (done or truncated):
            print(f"State: {state}")
            action = self.agent.select_action(state, is_eval=True)
            print(f"Action: {action}")
            activations = self.agent.policy_net.get_activations()
            print(f"Activations: {activations}")

            next_state, reward, done, truncated, _ = self.env.step(action)
            
            total_reward += reward
            state = next_state

        print(f"Reward: {total_reward}")

if __name__ == "__main__":
    seed = 42
    env_name = "CartPole-v1"
    if len(sys.argv) == 2:
        env_name = sys.argv[1]

    interpreter = RLInterpreter(env_name=env_name, seed=seed)
    # interpreter.interpret()
    # interpreter.check_agent_outputs()
    # interpreter.create_heatmap_pos_vel()
    # interpreter.create_heatmap_pole_angle_vel()
    interpreter.create_heatmap_vel_angle_vel()
