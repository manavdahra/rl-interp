import sys
import gymnasium as gym
from cartpole.agent import Agent

class Evaluator:
    def __init__(self, env_name, seed, episodes=10):
        self.env = gym.make(env_name, render_mode="human")
        self.seed = seed
        self.episodes = episodes
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.agent = Agent(state_size, action_size, restore=True, data_dir=f"data_dir/{env_name}")
    
    def evaluate(self):
        for _ in range(self.episodes):
            state, _ = self.env.reset(seed=self.seed)
            done = False
            truncated = False
            total_reward = 0
            
            while not (done or truncated):
                action = self.agent.select_action(state, is_eval=True)
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                state = next_state
                total_reward += reward
        
        self.env.close()
        print("Finished playing")
        
if __name__ == "__main__":
    seed = 42
    env_name = "CartPole-v1"
    if len(sys.argv) == 2:
        env_name = sys.argv[1]

    evaluator = Evaluator(env_name=env_name, seed=seed)
    evaluator.evaluate()
