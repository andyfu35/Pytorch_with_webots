import os
from controller import Supervisor

# 创建Supervisor实例
supervisor = Supervisor()

# 获取当前世界文件的路径
world_path = supervisor.getWorldPath()

# 加载新的世界文件
new_world_path = "path/to/your/new/world_file.wbt"
supervisor.simulationSetMode(supervisor.SIMULATION_MODE_PAUSE)
supervisor.worldReload()
supervisor.simulationResetPhysics()
supervisor.simulationReset()
supervisor.simulationSetMode(supervisor.SIMULATION_MODE_RUN)

# 等待一段时间以确保新的世界被加载
supervisor.step(100)

# 重新加载后的操作
# 这里可以添加一些你希望在重新加载后执行的代码

# 关闭仿真
supervisor.simulationQuit()
# import sys
# sys.path.append("C:/Program Files/Webots/lib/python")
