import os
from controller import Supervisor

# 创建Supervisor实例
supervisor = Supervisor()

# 获取当前世界文件的路径
world_path = supervisor.getWorldPath()

# 加载新的世界文件
new_world_path = r"C:\Users\andyf\PycharmProjects\Pytorch_with_webots\worlds\FuckingWorld.wbt"
supervisor.simulationSetMode(0)

# 检查新世界文件是否存在
if os.path.exists(new_world_path):
    # 加载新世界文件
    supervisor.worldLoad(new_world_path)
    supervisor.simulationResetPhysics()
    supervisor.simulationReset()
    supervisor.simulationSetMode(1)  # 使用数字值来表示运行模式

    # 等待一段时间以确保新的世界被加载
    supervisor.step(100)

    # 重新加载后的操作
    # 这里可以添加一些你希望在重新加载后执行的代码
else:
    print(f"指定的新世界文件 '{new_world_path}' 不存在。")

# 关闭仿真
supervisor.simulationQuit(1)
