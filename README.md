# RL Algorithms

This repository contains from-scratch implementations of reinforcement learning algorithms using PyTorch and Gymnasium, with a focus on understanding algorithms deeply rather than using high-level libraries.

Currently included:
- REINFORCE Policy Gradient on CartPole-v1
- Actor–Critic with Self-Play on a custom Tic-Tac-Toe environment

⸻

## CartPole-v1 — REINFORCE Policy Gradient

This project demonstrates solving CartPole-v1 using the REINFORCE (Monte-Carlo Policy Gradient) algorithm.

Highlights
- Pure policy-gradient method (no value function)
- Episode-level return computation
- Trained until the environment is considered solved

Algorithm

REINFORCE updates the policy in the direction of higher expected return:

$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t$


Run

```python train_cartpole.py```

Inference

```python cartpoleinference.py```

Results
- Typically solves CartPole-v1 within 500–1500 episodes
- Average reward ≥ 195
- Trained model weights (.pth) are included in the repository

⸻

## Tic-Tac-Toe — Actor–Critic with Self-Play

This project implements a custom Tic-Tac-Toe environment and trains an Actor–Critic agent using self-play against a frozen opponent.

What’s special here
- Custom environment written from scratch (tictactoe_env.py)
- Actor–Critic architecture:
- Actor: learns the policy
- Critic: learns the value function
- Self-play using an older frozen version of the agent
- Human-vs-agent evaluation

Files
- Environment: tictactoe_env.py
- Training: tictactoe_actorcritic.py
- Evaluation (human vs agent): tictactoe_eval.py
- Saved weights: actor and critic .pth files included

Train

```python tictactoe_actorcritic.py```

Evaluate (Human vs Agent)

```python tictactoe_eval.py```

The starting player is randomized so the agent must play both first and second.

Outcome
- Agent learns optimal or near-optimal Tic-Tac-Toe play
- Avoids illegal moves
- Consistently beats random and weak opponents
- Demonstrates stable learning via self-play

⸻

Requirements
- Python ≥ 3.8
- PyTorch
- Gymnasium
- NumPy

⸻

Goal of This Repository

This repo is meant to be a learning-first RL codebase, showing:
- How RL algorithms actually work internally
- How to write custom environments
- How to train, evaluate, and debug agents without shortcuts

More algorithms (PPO, DQN, multi-agent self-play) will be added over time.
