from controller import Supervisor
import math
import numpy as np

#
# supervisor.simulationReset()


class Control:
    def __init__(self, supervisor, robot_node):
        self.supervisor = supervisor
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.robot_node = robot_node
        self.position = self.robot_node.getPosition()
        self.motor_names = ['Joint_11', 'Joint_12', 'Joint_13', 'Joint_14',
                            'Joint_21', 'Joint_22', 'Joint_23', 'Joint_24']
        self.motors = [self.supervisor.getDevice(name) for name in self.motor_names]
        self.sensors = [self.supervisor.getDevice(name+"_sensor") for name in self.motor_names]
        for sensor in self.sensors:
            sensor.enable(self.timestep)
        self.joint_value = np.zeros((len(self.motors), 1))
        self.min_position = np.round(np.array(-math.pi)/2, 4)
        self.max_position = np.round(np.array(math.pi)/2, 4)
        self.joint_3_min_position = 0
        self.joint_3_max_position = np.round(np.array(-math.pi) * 3 / 4, 4)
        self.min_velocity = 0.0
        self.max_velocity = 3.0

    def sensor_value(self):
        for i, sensor in enumerate(self.sensors):
            self.joint_value[i] = sensor.getValue()
        # print(np.round(self.joint_value.flatten(), 3))
        return np.round(self.joint_value.flatten(), 3)

    def current_position(self):
        # print(np.round(np.array(self.position), 1))
        return np.round(np.array(self.position), 1)

    def next_action(self, action_instruction):
        for i, motor in enumerate(self.motors):
            # if i == 3:
            #     motor_position = np.interp(action_instruction[2 * i], [-1, 1], [self.joint_3_min_position, self.joint_3_max_position])
            # else:
            #     motor_position = np.interp(action_instruction[2*i], [-1, 1], [self.min_position, self.max_position])
            motor_position = np.interp(action_instruction[2 * i], [-1, 1], [self.min_position, self.max_position])
            motor_velocity = np.interp(action_instruction[2*i+1], [-1, 1], [self.min_velocity, self.max_velocity])
            motor.setPosition(motor_position)
            motor.setVelocity(motor_velocity)


if __name__ == '__main__':
    supervisor_ = Supervisor()
    while supervisor_.step(64) != -1:
        robot_node_ = supervisor_.getFromDef('Severus_node')
        ctrl = Control(supervisor_, robot_node_)
        ctrl.next_action([180, 1.0, 180, 1.0, 0, 0, 0, 0])
