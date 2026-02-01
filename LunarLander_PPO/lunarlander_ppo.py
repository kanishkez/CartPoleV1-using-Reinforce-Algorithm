import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

ENV_NAME = "LunarLander-v3"
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4

STEPS_PER_UPDATE = 2048
EPOCHS = 10
MINIBATCH_SIZE = 64

VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

MAX_EPISODES = 2000
SAVE_EVERY = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

def compute_gae(rewards, values, dones, next_value):
    advantages = []
    gae = 0
    values = values + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

def train():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    state, _ = env.reset()
    episode_reward = 0
    episode = 0

    while episode < MAX_EPISODES:
        states, actions, log_probs = [], [], []
        rewards, dones, values = [], [], []

        for _ in range(STEPS_PER_UPDATE):
            state_tensor = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            logits, value = model(state_tensor)

            dist = Categorical(logits=logits)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            dones.append(done)
            values.append(value.squeeze().item())

            state = next_state
            episode_reward += reward

            if done:
                print(f"Episode {episode} | Reward: {episode_reward:.1f}")
                state, _ = env.reset()
                episode_reward = 0
                episode += 1

                if episode % SAVE_EVERY == 0:
                    torch.save(model.state_dict(), "ppo_lunarlander.pt")
                    print("Model saved")

                if episode >= MAX_EPISODES:
                    break

        with torch.no_grad():
            next_state_tensor = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            _, next_value = model(next_state_tensor)
            next_value = next_value.item()

        advantages, returns = compute_gae(rewards, values, dones, next_value)

        states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        actions = torch.stack(actions).to(DEVICE)
        old_log_probs = torch.stack(log_probs).detach().to(DEVICE)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(EPOCHS):
            idx = np.random.permutation(len(states))
            for start in range(0, len(states), MINIBATCH_SIZE):
                batch_idx = idx[start:start + MINIBATCH_SIZE]

                logits, values_pred = model(states[batch_idx])
                dist = Categorical(logits=logits)

                new_log_probs = dist.log_prob(actions[batch_idx])
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - old_log_probs[batch_idx]).exp()

                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages[batch_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (returns[batch_idx] - values_pred.squeeze()).pow(2).mean()

                loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

    torch.save(model.state_dict(), "ppo_lunarlander_final.pt")
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    train()
