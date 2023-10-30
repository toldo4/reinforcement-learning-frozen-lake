import numpy as np
import gym


def random_agent(env, observation):
    return env.action_space.sample()


# Random agent evaluation
def evaluate_random_agent(env, num_episodes):
    total_reward = 0
    for _ in range(num_episodes):
        observation = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = random_agent(env, observation)
            observation, reward, done, _, _ = env.step(action)
            episode_reward += reward
        total_reward += episode_reward

    return total_reward / num_episodes


# Value Iteration to obtain the value function
def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            V_temp = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for prob, next_state, reward, done in env.P[s][a]:
                    V_temp[a] += prob * (reward + gamma * V[next_state])
            V[s] = np.max(V_temp)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V


# Obtaining the optimal policy from the value function
def extract_policy(env, V, gamma=0.99):
    policy = np.zeros(env.observation_space.n, dtype=int)
    for s in range(env.observation_space.n):
        action_values = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, _ in env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])
        policy[s] = np.argmax(action_values)
    return policy


# Policy evaluation
def evaluate_policy(env, num_episodes, policy):
    total_reward = 0
    for _ in range(num_episodes):
        observation, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = policy[observation]
            observation, _, reward, done, _ = env.step(action)
            episode_reward += reward
        total_reward += episode_reward

    return total_reward / num_episodes


def training(env, num_episodes):
    # Evaluate random agent
    avg_reward_random = evaluate_random_agent(env, num_episodes)
    print(f"Recompensa média do agente aleatório: {avg_reward_random}")

    # Get value function and optimal policy
    V = value_iteration(env)
    optimal_policy = extract_policy(env, V)

    # Evaluate optimal policy
    avg_reward_optimal = evaluate_policy(env, num_episodes, optimal_policy)
    print(f"Recompensa média da política ótima: {avg_reward_optimal}")
    return avg_reward_random, optimal_policy, avg_reward_optimal


def demo_agent(env, policy, num_episodes=1):
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

    avg_reward_random, optimal_policy, avg_reward_optimal = training(env, num_episodes)

    visual_env = gym.make('FrozenLake-v1', render_mode='human')
    demo_agent(visual_env, optimal_policy, 3)

if __name__ == '__main__':
    main()
