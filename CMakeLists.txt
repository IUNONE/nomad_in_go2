# see: https://docs.ros.org/en/humble/How-To-Guides/Ament-CMake-Python-Documentation.html#using-ament-cmake-python
# example: https://sachinkum0009.medium.com/create-ros2-pkg-for-c-and-python-files-26e95f9bc2c7
#

cmake_minimum_required(VERSION 3.5)
project(planner_in_go2)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

#---------------------------cpp-----------------------------------

include_directories(include include/common include/nlohmann)
link_directories(src)

# set dependency list variable for convenience
set (
  DEPENDENCY_LIST
  unitree_go
  unitree_api
  rclcpp
  std_msgs
  nav_msgs
  rosbag2_cpp
  tf2_ros
  geometry_msgs
)

# msg interface
# rosidl_generate_interfaces(${PROJECT_NAME}
#   "msg/waypoints.msg"
# )

# find dependencies for cpp
find_package(ament_cmake REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_api REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)

# cpp executables
# add_executable(joy_stick src/joy_stick.cpp)
add_executable(follow_traj src/follow_traj.cpp src/common/ros2_sport_client.cpp)
add_executable(ego_state src/ego_state.cpp)

# create executable nodes & dependencies
# usage: ament_target_dependencies(<executable-name> [dependencies])
# ament_target_dependencies(joy_stick ${DEPENDENCY_LIST})
ament_target_dependencies(follow_traj ${DEPENDENCY_LIST})
ament_target_dependencies(ego_state ${DEPENDENCY_LIST})

# install nodes
install(
  TARGETS 
    # custom_joy_stick 
    follow_traj
    ego_state
  DESTINATION lib/${PROJECT_NAME}
)   


#---------------------------py--------------------------------------
# ros2 dependencies & make tool
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# install python modules used in scripts
ament_python_install_package(model)
ament_python_install_package(utils)

# install your py scripts module
install(
  PROGRAMS
    scripts/goal_tf.py
    scripts/img_puber.py
    scripts/planner.py
  DESTINATION lib/${PROJECT_NAME}
)

# install launch files
install(
  DIRECTORY launch
  DESTINATION lib/${PROJECT_NAME}
)

#--------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
