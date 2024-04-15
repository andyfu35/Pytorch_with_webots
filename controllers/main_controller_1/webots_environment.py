from controller import Supervisor
from robot_control import Control
import numpy as np
import math

supervisor = Supervisor()


class EnvironmentCtrl:
    def __init__(self):
        self.robot = "Severus_node"
        self.robot_node = supervisor.getFromDef(self.robot)
        self.target_position = [0.0, 7.0, 0.0]
        self.ctrl = Control(supervisor, self.robot_node)
        self.previous_position = self.robot_node.getPosition()
        self.position = self.robot_node.getPosition()
        self.max_steps = 200
        self.current_step = 0

    def reset_environment(self):
        supervisor.simulationReset()
        supervisor.step(0)
        self.current_step = 0
        self.previous_position = self.robot_node.getPosition()  # 重置前一位置
        self.position = self.robot_node.getPosition()
        return self.get_state()

    def update_position(self):
        self.previous_position = self.position
        self.position = self.robot_node.getPosition()

    def calculate_velocity(self):
        return np.linalg.norm(np.array(self.position) - np.array(self.previous_position)) / supervisor.getBasicTimeStep()

    def get_state(self):
        self.update_position()
        state = np.round(np.array(self.position), 3)
        # print(f"State obtained: {state}")
        return state

    def check_condition(self, i):
        self.update_position()
        return self.position[i] >= self.target_position[i]

    def goal_achieved(self):
        result = all(self.check_condition(_) for _ in range(3))
        return result

    def calculate_reward(self):
        current_distance = np.linalg.norm(np.array(self.target_position) - np.array(self.position))
        previous_distance = np.linalg.norm(np.array(self.target_position) - np.array(self.previous_position))
        velocity = self.calculate_velocity()

        if current_distance < previous_distance:
            reward = velocity * 10 / (current_distance + 0.1)
        else:
            reward = -velocity * 5 / (current_distance + 0.1)
            
        return reward

    def step(self, action_instruction):
        supervisor.step(int(supervisor.getBasicTimeStep()))
        self.current_step += 1

        self.ctrl.next_action(action_instruction)
        new_state = self.get_state()
        reward = self.calculate_reward()
        done = self.goal_achieved() or self.current_step >= self.max_steps
        if done:
            self.reset_environment()
        print(new_state, reward, done)
        return new_state, reward, done


if __name__ == '__main__':
    while supervisor.step(64) != -1:
        EnvCtrl = EnvironmentCtrl()
        EnvCtrl.step([0, 0, 1.0, 1.0, 0, 0, 0, 0])
        # EnvCtrl.get_state()
