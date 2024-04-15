import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from environmental_controller import EnvironmentCtrl
from robot_controller import Control


# 定義DQN模型
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# 定義重播緩衝區（Replay Buffer）
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# 參數設定
state_dim = 3  # 狀態維度示例
action_dim = 5  # 行動維度示例
hidden_dim = 64  # 隱藏層維度示例
capacity = 10000  # 重播緩衝區容量
batch_size = 64  # 訓練批次大小
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

# ε-greedy策略
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)

# 創建DQN和重播緩衝區實例
dqn_model = DQN(state_dim, action_dim, hidden_dim)
replay_buffer = ReplayBuffer(capacity)
optimizer = optim.Adam(dqn_model.parameters())

# 使用ε-greedy選擇行動的函數
def select_action(state, epsilon):
    if random.random() > epsilon:
        state = torch.FloatTensor(state).unsqueeze(0)
        q_value = dqn_model(state)
        action = q_value.max(1)[1].item()
    else:
        action = random.randrange(action_dim)
    return action

# 執行訓練的函數
def train(replay_buffer, model, optimizer, batch_size):
    if len(replay_buffer) < batch_size:
        return
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(done)

    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + 0.99 * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.data).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 初始化您的環境和控制器
env = EnvironmentCtrl(angle_instruction=[0]*16, velocity_instruction=[1]*16)
control = Control(env.supervisor, env.robot_node, angle_instruction=[0]*16, velocity_instruction=[1]*16)

# 主訓練循環
for frame_idx in range(1000):
    epsilon = epsilon_by_frame(frame_idx)
    state = env.get_state()  # 從您的環境獲取當前狀態
    action = select_action(state, epsilon)  # 基於當前狀態決定行動
    env.ctrl.apply_action(action)  # 將行動應用到環境
    next_state, reward, done = env.step()  # 執行步驟並觀察新的狀態和獎勵
    replay_buffer.push(state, action, reward, next_state, done)  # 將經驗存儲到緩衝區
    train(replay_buffer, dqn_model, optimizer, batch_size)  # 訓練模型

    if done:
        env.reset()  # 如果完成則重置環境
P