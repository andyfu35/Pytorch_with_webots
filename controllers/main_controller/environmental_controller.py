from controller import Supervisor
from robot_controller import Control


supervisor = Supervisor()


class EnvironmentCtrl:
    def __init__(self):
        super().__init__()
        self.target_position = [0, 0, 0]

    @staticmethod
    def reset_environment():
        supervisor.simulationReset()

    def get_state(self):
        # 獲取狀態信息，例如機器人的位置和距離感測器的讀數
        position = self.robot_node.getPosition()
        distance = self.distance_sensor.getValue()
        return position + (distance,)

    def calculate_reward(self):
        # 根據機器人的當前狀態計算獎勵
        position = self.robot_node.getPosition()
        if self.goal_achieved(position):
            return 100  # 達到目標的獎勵
        else:
            return -1  # 每一步的小懲罰，鼓勵快速達成目標

    def goal_achieved(self, position):
        # 判斷是否達到目標
        return all(abs(position[i] - self.target_position[i]) < 0.1 for i in range(3))

    def step(self, action):
        # 綜合步驟：執行動作，返回新的狀態、獎勵和是否結束
        self.perform_action(action)
        supervisor.step(int(supervisor.getBasicTimeStep()))
        new_state = self.get_state()
        reward = self.calculate_reward()
        done = self.goal_achieved(new_state[:3])  # 假設狀態的前三個元素是位置
        return new_state, reward, done