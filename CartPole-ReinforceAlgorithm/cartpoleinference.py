import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x


env = gym.make("CartPole-v1", render_mode="human")

INPUT_DIM = env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = env.action_space.n
DROPOUT = 0.0

policy = PolicyNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)
policy.load_state_dict(torch.load("cartpole_policy.pth"))
policy.eval()  # IMPORTANT

obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

    with torch.no_grad():
        logits = policy(obs_tensor)
        probs = F.softmax(logits, dim=-1)
        action = torch.argmax(probs).item()  # deterministic action

    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

print("Total reward:", total_reward)
env.close()
