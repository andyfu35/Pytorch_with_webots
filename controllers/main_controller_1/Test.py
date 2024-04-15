import torch
import torch.nn as nn
import torch.optim as optim

class SimpleActor(nn.Module):
    def __init__(self, state_size, action_size):
        super(SimpleActor, self).__init__()
        self.fc = nn.Linear(state_size, action_size)
        self.activation = nn.Tanh()  # Ensure the action output is between -1 and 1

    def forward(self, state):
        return self.activation(self.fc(state))

class SimpleCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(SimpleCritic, self).__init__()
        self.state_path = nn.Linear(state_size, 128)
        self.action_path = nn.Linear(action_size, 128)
        self.out = nn.Linear(256, 1)

    def forward(self, state, action):
        state_out = torch.relu(self.state_path(state))
        action_out = torch.relu(self.action_path(action))
        concat = torch.cat((state_out, action_out), dim=1)
        return self.out(concat)

# Example dimensions for state and action
state_size = 10
action_size = 4

# Initialize actor and critic
actor = SimpleActor(state_size, action_size)
critic = SimpleCritic(state_size, action_size)

# Define the optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

# Create some example data to pass through the models
state = torch.randn(1, state_size)
action = actor(state)
critic_output = critic(state, action.detach())  # Detach the action to stop gradients flowing into the actor network

print("Actor Output:", action)
print("Critic Output:", critic_output)

# 假设已经有以下的模块导入和模型初始化
# import torch
# import torch.nn as nn
# import torch.optim as optim

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移到指定的设备
actor.to(device)
critic.to(device)

# 创建优化器
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

# 创建假数据来模拟环境的反馈
state = torch.randn(1, state_size).to(device)
action = actor(state)
reward = torch.tensor([[1.0]]).to(device)  # 假设得到的奖励
target_value = reward  # 简化例子中直接使用奖励作为目标值

# 前向传播以获取当前动作的值
predicted_value = critic(state, action.detach())

# 计算批评家的损失
value_loss = torch.nn.functional.mse_loss(predicted_value, target_value)

# 反向传播更新批评家
critic_optimizer.zero_grad()
value_loss.backward()
critic_optimizer.step()

# 计算演员的损失
policy_loss = -critic(state, actor(state)).mean()

# 反向传播更新演员
actor_optimizer.zero_grad()
policy_loss.backward()
actor_optimizer.step()

# 打印梯度和损失信息
print("Value Loss:", value_loss.item())
print("Policy Loss:", policy_loss.item())
