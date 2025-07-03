import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_motion_planning import *

# Create environment with custom obstacles
grid_env = Grid(51, 31)
obstacles = grid_env.obstacles
for i in range(10, 21):
    obstacles.add((i, 15))
for i in range(15):
    obstacles.add((20, i))
for i in range(15, 30):
    obstacles.add((30, i))
for i in range(16):
    obstacles.add((40, i))
grid_env.update(obstacles)

# Create APF planner
plt = APF(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
plt.run()
