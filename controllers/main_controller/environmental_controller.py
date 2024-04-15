from controller import Supervisor
from robot_controller import Control
import numpy as np
import math

supervisor = Supervisor()


class EnvironmentCtrl:
    def __init__(self, angle_instruction_, velocity_instruction_):
        # super().__init__()
        self.robot = "Severus_node"
        self.robot_node = supervisor.getFromDef(self.robot)
        self.target_position = [0.0, 7.0, 0.0]
        self.ctrl = Control(supervisor, self.robot_node, angle_instruction_, velocity_instruction_)

        self.position = self.robot_node.getPosition()
        self.max_steps = 10000
        self.current_step = 0

    @staticmethod
    def reset_environment():
        supervisor.simulationReset()

    def update_position(self):
        self.position = self.robot_node.getPosition()

    def get_state(self):
        self.update_position()
        return np.round(np.array(self.position), 1)

    def check_condition(self, i):
        self.update_position()
        return self.position[i] >= self.target_position[i]

    def goal_achieved(self):
        result = all(self.check_condition(_) for _ in range(3))
        return result

    def calculate_reward(self):
        if self.goal_achieved():
            return 1000
        self.update_position()
        distance = math.sqrt(sum((cp - tp) ** 2 for cp, tp in zip(self.get_state(), self.target_position)))
        if self.get_state()[2] < 0 or distance > 100:
            return -1
        reward = 1 / (distance + 0.001)
        return reward

    def step(self):
        supervisor.step(int(supervisor.getBasicTimeStep()))
        self.current_step += 1

        self.ctrl.run()
        new_state = self.get_state()
        reward = self.calculate_reward()
        done = self.goal_achieved() or self.current_step >= self.max_steps
        if done:
            self.reset_environment()
        print(new_state, reward, done)
        return new_state, reward, done


if __name__ == "__main__":
    # angle_instruction = [180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 180]
    # velocity_instruction = [1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    angle_instruction = [180, 0, 0, 0]
    velocity_instruction = [1.0, 0, 0, 0]
    Env = EnvironmentCtrl(angle_instruction, velocity_instruction)
    while supervisor.step(int(supervisor.getBasicTimeStep())) != -1:
        Env.step()
