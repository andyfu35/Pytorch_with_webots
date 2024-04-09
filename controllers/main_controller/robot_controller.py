from controller import Supervisor
import math
import numpy as np

supervisor = Supervisor()
supervisor.simulationReset()


class Control:
    def __init__(self, robot_node, motor_number, angle_instructions, velocity_instruction):
        self.timestep = 64
        self.robot_node = supervisor.getFromDef(str(robot_node))
        self.angle_instructions = angle_instructions
        self.velocity_instruction = velocity_instruction
        self.motor_number = motor_number
        self.position = self.robot_node.getPosition()
        self.motor_names = [['Joint-11', 'Joint-12', 'Joint-13', 'Joint-14'],
                            ['Joint-21', 'Joint-22', 'Joint-23', 'Joint-24'],
                            ['Joint-31', 'Joint-32', 'Joint-33', 'Joint-34'],
                            ['Joint-41', 'Joint-42', 'Joint-43', 'Joint-44']]
        self.motors = [supervisor.getDevice(name) for name in self.motor_names[self.motor_number-1]]
        self.sensors = [supervisor.getDevice(name+"_sensor") for name in self.motor_names[self.motor_number-1]]
        self.joint_value = [0, 0, 0, 0]
        self.t = 0
        for sensor in self.sensors:
            sensor.enable(self.timestep)

    @staticmethod
    def angle_conversion(angle):
        return angle / 180 * math.pi

    @staticmethod
    def diameter_conversion(diameter):
        return round(diameter / math.pi * 180, 0)

    def sensor_value(self):
        for i, sensor in enumerate(self.sensors):
            self.joint_value[i] = sensor.getValue()
        print(np.round(np.array(self.joint_value), 3))
        return np.round(np.array(self.joint_value), 3)

    def current_position(self):
        print(np.round(np.array(self.position), 1))
        return np.round(np.array(self.position), 1)

    def run(self):
        for i, motor in enumerate(self.motors):
            motor.setPosition(self.angle_conversion(self.angle_instructions[i]))
            motor.setVelocity(self.velocity_instruction[i])


if __name__ == "__main__":
    while supervisor.step(64) != -1:
        ctrl_1 = Control('Node_base', 1, [0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0])
        ctrl_1.run()
        ctrl_1.sensor_value()
        ctrl_1.current_position()
