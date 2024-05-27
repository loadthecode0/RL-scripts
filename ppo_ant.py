import glfw
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Initialize GLFW to catch any initialization issues early
if not glfw.init():
    raise Exception("GLFW can't be initialized")

# Create environment
env = gym.make("Ant-v4", render_mode="human")

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=2)  # Increased verbosity
# Train the agent and display a progress bar
model.learn(total_timesteps=int(2e5), progress_bar=True)
# Save the agent
model.save("models/ppo_ant")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = PPO.load("models/ppo_ant", env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward} +/- {std_reward}")


# Enjoy trained agent
vec_env = model.get_env()

obs = vec_env.reset()
if isinstance(obs, tuple):
    obs = obs[0]

for i in range(1000):
    print(i)
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)  # Expect 4 values
    vec_env.render()  # Comment this line out to test without rendering
    if dones:
        obs = vec_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

# Close the environment
vec_env.close()

# Terminate GLFW
glfw.terminate()
