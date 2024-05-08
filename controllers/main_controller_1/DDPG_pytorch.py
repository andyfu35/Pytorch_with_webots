import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from webots_environment import EnvironmentCtrl

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_size),
            nn.Tanh()
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_path = nn.Sequential(
            nn.Linear(state_size, 400),
            nn.ReLU()
        )
        self.action_path = nn.Sequential(
            nn.Linear(action_size, 300),
            nn.ReLU()
        )
        self.out = nn.Linear(700, 1)

    def forward(self, state, action):
        state_out = self.state_path(state)
        action_out = self.action_path(action)
        concat = torch.cat((state_out, action_out), dim=1)
        output = self.out(concat)
        # print(f"Input to linear layer: {concat.shape}, Output from linear layer: {output.shape}")
        return output


def select_action(state, actor, noise_level):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = actor(state).detach().cpu().numpy().flatten()
    noisy_action = np.clip(action + np.random.normal(0, noise_level, size=action.shape), -1, 1)
    return noisy_action


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.memory, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.memory)


def update_model(memory, actor, critic, actor_optimizer, critic_optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return

    # 获取数据并准备张量
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    states = torch.FloatTensor(states).to(device)
    actions = torch.FloatTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device).view(-1, 1)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device).view(-1, 1)

    # 计算Critic的损失
    current_q_values = critic(states, actions)
    next_actions = actor(next_states).detach()  # 使用detach，确保没有梯度传播
    next_q_values = critic(next_states, next_actions)
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)
    critic_loss = nn.MSELoss()(current_q_values, expected_q_values.detach())

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # 计算Actor的损失
    actor_optimizer.zero_grad()
    policy_loss = -critic(states, actor(states)).mean()
    policy_loss.backward()
    actor_optimizer.step()



def train_ddpg():
    env = EnvironmentCtrl()
    state_dim = env.get_state().shape[0]
    action_dim = 16  # 为四个电机的位置和速度
    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=0.0001)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)
    memory = ReplayMemory(100000)
    episodes = 1000
    batch_size = 64
    gamma = 0.99

    for episode in range(episodes):
        state = env.reset_environment()
        print(f"Initial state shape: {state.shape}")
        episode_reward = 0
        while True:
            action = select_action(state, actor, 0.1)
            next_state, reward, done = env.step(action)
            memory.push(state, action, reward, next_state, done)
            update_model(memory, actor, critic, actor_optimizer, critic_optimizer, batch_size, gamma)
            state = next_state
            episode_reward += reward
            if done:
                break
        print(f"Episode {episode} Reward: {episode_reward}")


if __name__ == "__main__":
    train_ddpg()
