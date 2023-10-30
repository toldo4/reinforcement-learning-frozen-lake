![FrozenLake Demo Start](./demo/demo_start.png)

# Reinforcement Learning with FrozenLake

This project aims to explore the basic concepts of Reinforcement Learning using the `FrozenLake` environment from the OpenAI Gym library.

## Concepts covered:

- Introduction to reinforcement learning: agents, environments, actions, rewards.
- Policies and value functions.
- Bellman's equation.

---

## GIF Comparison

See each agent in action (all trained for 10,000 episodes):

| Random Agent | Random with Bellman |
|:------------:|:-------------------:|
| ![Random Agent GIF](./demo/random_agent.gif) | ![Random Bellman GIF](./demo/random_bellman.gif) |

| Q-Learning Agent | SARSA Agent |
|:----------------:|:-----------:|
| ![Q-Learning GIF](./demo/qlearning.gif) | ![SARSA GIF](./demo/sarsa.gif) |

#### __* the best performance after 10000 episodes was from Q-Learning Agent__

---

## Instructions:

### Environment Setup:

It's recommended to create a virtual environment to install the necessary dependencies and maintain the project's consistency:

```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

After activating the virtual environment, install the dependencies through the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Execution:

There are four main scripts to run:
- `random_agent.py`: Initial random agent implementation.
- `random_agent_bellman_function.py`: Random agent implementation with Bellman's function.
- `qlearning_agent.py`: Agent implemented using the Q-Learning algorithm.
- `sarsa_agent.py`: Agent implemented using the SARSA algorithm.

To run any of these scripts, use:

```bash
python <script_name>.py
```

In addition, the project also contains an auxiliary script test_pygame.py that can be used to validate the installation of pygame:

```bash
python test_pygame.py
```

---

## Implemented Algorithms:

### Q-Learning:

A reinforcement learning technique where the agent learns to act in a way that maximizes the expected reward over time.

### SARSA (State-Action-Reward-State-Action):

An on-policy control technique where the agent learns to evaluate actions in the environment based on actual rewards received.

---

## Environment:

### FrozenLake (Stochastic):

The agent must navigate a frozen lake and reach the goal without falling into holes. In stochastic mode, there's a chance the agent might slip even when given a clear movement instruction.

---

## Credits:

This project was developed with the help of the OpenAI platform and based on tutorials and documentation from the OpenAI Gym library.
