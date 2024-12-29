import sys
import gymnasium as gym
from cartpole.agent import Agent

class Trainer:
    def __init__(self, env_name, seed, episodes=500, target_update_freq=10):
        self.env = gym.make(env_name)
        self.seed = seed
        self.episodes = episodes
        self.target_update_freq = target_update_freq
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.agent = Agent(state_size, action_size, data_dir=f"data_dir/{env_name}")
    
    def train(self):
        for episode in range(self.episodes):
            state, _ = self.env.reset(seed=self.seed)  # Gymnasium returns (obs, info)
            total_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.agent.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)  # Gymnasium returns (obs, reward, terminated, truncated, info)
                
                self.agent.memory.push(state, action, reward, next_state, done)
                self.agent.train()
                
                state = next_state
                total_reward += reward
            
            self.agent.update_target_network()
                
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.agent.epsilon:.2f}")
                
        print("Finished training model")
        self.env.close()
        self.agent.save_agent()
        return self.agent
        
if __name__ == "__main__":
    seed = 42
    env_name = "CartPole-v1"
    if len(sys.argv) == 2:
        env_name = sys.argv[1]

    trainer = Trainer(env_name=env_name, seed=seed)
    trainer.train()