from environmental_controller import EnvironmentCtrl
from DQN_pytorch import DQN, ReplayMemory

env = EnvironmentCtrl(robot_node)

policy_net = DQN(env.input_size, env.output_size).to(device)
target_net = DQN(env.input_size, env.output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
