from controller import Supervisor

supervisor = Supervisor()


class EnvironmentCtrl:
    def __init__(self, robot_node):
        self.robot_node = robot_node

    @staticmethod
    def reset_environment():
        supervisor.simulationReset()

    def robot_position(self):
        position = self.robot_node.getPosition()
        return position

    def goal_achieved(self):
        target_position = [0, 0, 0]
        axis = ['X', 'Y', 'Z']
        for i in range(3):
            if self.robot_position()[i] >= target_position[i]:
                print(axis[i], ' achieved')
