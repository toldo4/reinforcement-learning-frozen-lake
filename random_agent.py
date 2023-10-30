import gym


def random_agent(env, observation):
    return env.action_space.sample()


def training(env, num_episodes):
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

    print(f"Recompensa média: {total_reward / num_episodes}")


def main():
    env = gym.make("FrozenLake-v1")
    num_episodes = 1000

    training(env, num_episodes)


if __name__ == '__main__':
    main()
