from controller import Supervisor
import math

supervisor = Supervisor()
supervisor.simulationReset()


class Control:
    def __init__(self, robot_node, angle_instructions, velocity_instruction):
        self.robot_node = supervisor.getFromDef(str(robot_node))
        self.angle_instructions = angle_instructions
        self.velocity_instruction = velocity_instruction
        self.motor_names = ['Joint-11', 'Joint-12', 'Joint-13', 'Joint-14']
        self.motors = [supervisor.getDevice(name) for name in self.motor_names]
        self.motor = supervisor.getDevice('Joint-11')

    @staticmethod
    def angle_conversion(angle):
        return angle / 180 * math.pi

    def run(self):
        # self.motor.setPosition(3.14)
        # self.motor.setVelocity(1.0)
        for i, motor in enumerate(self.motors):
            print(self.angle_conversion(self.angle_instructions[i]))
            motor.setPosition(self.angle_conversion(self.angle_instructions[i]))
            motor.setVelocity(self.velocity_instruction[i])


if __name__ == "__main__":
    while supervisor.step(64) != -1:
        ctrl = Control('Node_base', [180, 180, 180, 180], [1.0, 2.0, 1.0, 1.0])
        ctrl.run()
