import vmas

# Create the environment
env = vmas.make_env(
    scenario="simple_adversary", # can be scenario name or BaseScenario class
    num_envs=32,
    device="cpu", # Or "cuda" for GPU
    continuous_actions=False,
    max_steps=None, # Defines the horizon. None is infinite horizon.
    seed=None, # Seed of the environment
    n_agents=3,  # Additional arguments you want to pass to the scenario
    display_info=True
)
# Reset it
obs = env.reset()

# Step it with deterministic actions (all agents take their maximum range action)
for _ in range(1000):
    obs, rews, dones, info = env.step(env.get_random_actions())
    env.render()
