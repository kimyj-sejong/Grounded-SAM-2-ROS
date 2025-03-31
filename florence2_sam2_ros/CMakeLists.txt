cmake_minimum_required(VERSION 3.0.2)
project(florence2_sam2_ros)

## Find required catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  std_msgs
)

## Declare catkin package (for dependency tracking)
catkin_package()

## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/florence2_sam2_ros_realsense_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Include directories (add any necessary include paths here)
include_directories(
  ${catkin_INCLUDE_DIRS}
)
