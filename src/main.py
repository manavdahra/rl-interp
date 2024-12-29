import sys
from cartpole.agent import Agent
import gymnasium as gym
import torch

class RLInterpreter:
    def __init__(self, env_name, seed):
        self.env_name = env_name
        self.seed = seed
        self.env = gym.make(env_name)
        
    def interpret(self):
        print("interpreting model weights and biases...")
        
        agent = Agent(4, 2)
        agent.load_agent()
        
        state, _ = self.env.reset(seed=self.seed)
        for _ in range(5):
            print(f"State: {state}")
            action = agent.select_action(state, is_eval=True)
            self.env.step(action)
            print(f"Action: {action}")


if __name__ == "__main__":
    seed = 42
    env_name = "CartPole-v1"
    if len(sys.argv) == 2:
        env_name = sys.argv[1]

    interpreter = RLInterpreter(env_name=env_name, seed=seed)
    interpreter.interpret()
