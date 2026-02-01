import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

ENV_NAME = "LunarLander-v3"
MODEL_PATH = "ppo_lunarlander_final.pt"
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

def evaluate(episodes=5):
    env = gym.make(ENV_NAME, render_mode="human")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                logits, _ = model(state_tensor)
                action = torch.argmax(logits).item()  # deterministic

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep} | Total Reward: {total_reward:.1f}")

    env.close()

if __name__ == "__main__":
    evaluate(episodes=10)
