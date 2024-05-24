# 1. File structure
```
unitree_ros2/
    ├── cyclonedds_ws
    ├── planner_in_go2
    └── unitree_ros2_setup.sh 
```

```
unitree_ros2/planner_in_go2/src
├── include/
├── launch/          # api request encoder
├── scripts/         # py node 
├── src/             # cpp node
├── model/           # py planner module
├── utils/           # py utils
├── CMakeLists.txt
└── package.xml
```

# 2. Make and install

```bash
# 1. build cyclonedds_ws, follow: https://support.unitree.com/home/zh/developer/ROS2_service

# 2. build lidar xt16 driver, follow https://github.com/HesaiTechnology/HesaiLidar_General_ROS/tree/ROS2

# 3. build customized planner
source unitree_ros2_setup.bash
cd planner_in_go2
colcon build
```


# 3. Run
NOTE: if you use conda, should first run `conda deactivate` for the python version problem 

```bash
cd ~/unitree_ros2/planner_in_go2
source ./install/local_setup.bash

ros2 run planner_in_go2 <NodeName>
python3 ./install/
```

run nodes from launch file :
```bash
ros2 launch planner_in_go2 MLplanner.launch.py # param1:=5 param2:=6
```

check whole the topic :
```bash
ros2 topic list
ros2 topic echo /MLplanner/xxx
```