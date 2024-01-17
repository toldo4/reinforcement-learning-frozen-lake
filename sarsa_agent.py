import numpy as np
import gymnasium as gym
#from gym.wrappers.monitoring.video_recorder import VideoRecorder
from tqdm import tqdm
import wandb

from PIL import Image, ImageDraw, ImageFont

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

direction_map = {0: "left", 1: "down", 2: "right", 3: "up"}

import os

hostname = os.uname()[1]


# Initialize wandb
wandb.init(
    project="FrozenLake",
    name = hostname + ":" + wandb.util.generate_id(),
    tags=["sarsa", ],
    config={
        "algorithm": "Sarsa",
        "timesteps": 100000,
        "env": "FrozenLakeEnv"
    }
)



def epsilon_greedy_policy(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(len(Q[state]))
    else:
        return np.argmax(Q[state])


def sarsa(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    pbar = tqdm(total=num_episodes, dynamic_ncols=True)
    for episode in range(num_episodes):
        state, _ = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon)
        done = False
        episode_reward = 0
        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)
            td_target = reward + gamma * Q[next_state, next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            state, action = next_state, next_action
            episode_reward += reward
        pbar.update(1)
        if episode % 1000 == 0:
            avg_reward = evaluate_policy(env, Q, 100)
            pbar.set_description(f"Average reward: {avg_reward:.2f}")
            wandb.log({"episode": episode, "avg_reward": avg_reward})
    pbar.close()
    return Q


def evaluate_policy(env, Q, num_episodes):
    total_reward = 0
    policy = np.argmax(Q, axis=1)
    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = policy[observation]
            observation, reward, done, _, _ = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / num_episodes


def demo_agent(env, Q, num_episodes=1):

    policy = np.argmax(Q, axis=1)
    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        print("\nDemo: Episode:", episode + 1)
        counter = 0
        ascii_string = ""
        while not done:
            counter += 1
            env.render()
            action = policy[observation]
            observation, reward, done, _, _ = env.step(action)
            ascii_string += f"Action {episode+1}-{counter}: {direction_map[action]}: reward: {reward}\n"
        env.render()
        print(ascii_string)

        img = Image.new("RGB", (500, 1000), color="beige")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), ascii_string, fill="brown")
        wandb.log({"ascii_image": wandb.Image(img)})



def main():
    env = gym.make("FrozenLake-v1", render_mode='ansi')
    num_episodes = 10000

    Q_sarsa = sarsa(env, num_episodes)
    avg_reward = evaluate_policy(env, Q_sarsa, num_episodes)
    print(f"Average reward after SARSA: {avg_reward}")

    visual_env = gym.make('FrozenLake-v1', render_mode='human')
    demo_agent(visual_env, Q_sarsa, 3)


if __name__ == '__main__':
    main()
