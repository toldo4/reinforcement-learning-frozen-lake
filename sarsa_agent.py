import numpy as np
import gym
from tqdm import tqdm


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
        print("\nEpisode:", episode + 1)
        while not done:
            env.render()
            action = policy[observation]
            observation, _, done, _, _ = env.step(action)
        env.render()


def main():
    env = gym.make("FrozenLake-v1")
    num_episodes = 10000

    Q_sarsa = sarsa(env, num_episodes)
    avg_reward = evaluate_policy(env, Q_sarsa, num_episodes)
    print(f"Average reward after SARSA: {avg_reward}")

    visual_env = gym.make('FrozenLake-v1', render_mode='human')
    demo_agent(visual_env, Q_sarsa, 3)


if __name__ == '__main__':
    main()
