from controller import Supervisor
import math
import numpy as np

# supervisor = Supervisor()
# supervisor.simulationReset()


class Control:
    def __init__(self, supervisor, robot_node, action_instruction):
        self.supervisor = supervisor
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.robot_node = robot_node
        self.angle_instructions = action_instruction[0]
        self.velocity_instruction = action_instruction[1]
        self.position = self.robot_node.getPosition()
        self.motor_names = ['Joint-11', 'Joint-12', 'Joint-13', 'Joint-14']
        self.motors = [self.supervisor.getDevice(name) for name in self.motor_names]
        self.sensors = [self.supervisor.getDevice(name+"_sensor") for name in self.motor_names]
        for sensor in self.sensors:
            sensor.enable(self.timestep)
        self.joint_value = np.zeros((len(self.motors), 1))

    def sensor_value(self):
        for i, sensor in enumerate(self.sensors):
            self.joint_value[i] = sensor.getValue()
        print(np.round(self.joint_value.flatten(), 3))
        return np.round(self.joint_value.flatten(), 3)

    def current_position(self):
        print(np.round(np.array(self.position), 1))
        return np.round(np.array(self.position), 1)

    def next_action(self):
        for i, motor in enumerate(self.motors):
            motor.setPosition(self.angle_instructions[i])
            motor.setVelocity(self.velocity_instruction[i])


# if __name__ == "__main__":
#     angle_instruction = [180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 180]
#     velocity_instruction = [1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]
#     ctrl_1 = Control('Node_base', angle_instruction, velocity_instruction)
#     while supervisor.step(ctrl_1.timestep) != -1:
#         ctrl_1.run()
#         ctrl_1.sensor_value()
#         ctrl_1.current_position()
