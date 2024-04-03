from controller import Supervisor
from time import sleep

supervisor = Supervisor()
supervisor.simulationReset()

robot_node = supervisor.getFromDef('Node_base')
motor = supervisor.getDevice('Joint-11')

# 检查是否成功获取了机器人节点
if robot_node is None:
    print('Failed')
else:
    print('Succeed')
    print(robot_node)

target_angle = 3.14
target_velocity = 1.0
i = 0

while supervisor.step(64) != -1:
    i = i + 0.1
    motor.setPosition(target_angle)
    motor.setVelocity(target_velocity)
    position = robot_node.getPosition()
    print('Robot position: %f %f %f\n' % (position[0], position[1], position[2]))
    if i >= 5:
        print('Achieve goal')
        supervisor.simulationReset()
        break

print('Simulation stop')
