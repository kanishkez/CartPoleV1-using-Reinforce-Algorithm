
# CartPole-v1 Solved with REINFORCE Policy Gradient

This repository demonstrates solving the **CartPole-v1** environment from the Gymnasium library using the **REINFORCE Policy Gradient** algorithm. The project includes a complete implementation in Python and PyTorch.

![cartpole](https://github.com/user-attachments/assets/9f6c694c-49f5-456f-8da7-c8722cb4d2d8)



## Overview

The goal of the **CartPole-v1** environment is to balance a pole on a moving cart by applying forces left or right. This project uses the **REINFORCE Policy Gradient** method, which is a Monte Carlo-based policy gradient algorithm to learn the optimal policy.  


## Algorithm

**REINFORCE** is a policy gradient method where the policy is updated in the direction of higher expected reward.  

Steps:  
1. Initialize a policy network with parameters θ.  
2. For each episode:  
   - Sample actions from the policy distribution.  
   - Collect rewards for the episode.  
   - Compute discounted returns.  
   - Update policy parameters θ using gradient ascent:  

$\[\theta \leftarrow \theta + \alpha \sum_{t} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t]\$

Where:  
- $\(G_t\)$ is the discounted return from time step $\(t\)$.  
- $\(\pi_\theta(a_t|s_t)\)$ is the policy probability of taking action $\(a_t\)$ in state $\(s_t\)$.


## Requirements

- Python >= 3.8  
- [PyTorch](https://pytorch.org/)  
- [Gymnasium](https://gymnasium.farama.org/) (or OpenAI Gym)  
- Numpy  
- Matplotlib (optional, for plotting reward curves)  

Usage

Run the training script:
```
python train_cartpole.py
```
Inference:
```
python cartpoleinference.py
```

Note:
Repository also contains the .pth file that contains the weights of the trained model.
Options:
- Modify hyperparameters like learning rate, gamma, or hidden layers in train_cartpole.py.
- Enable plotting of rewards for monitoring convergence.


Results
- The model typically solves CartPole-v1 within 500–1500 episodes.
- The average reward per episode converges to ≥ 195, which is considered solved according to OpenAI Gym standards.
- Example training curve:
