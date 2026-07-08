cmake_minimum_required(VERSION 3.8)
project(graspsam_ros2)

# Required
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)


# Interfaces
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Grasp.msg"
  "srv/RunGraspSAM.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# -------------------------
# Install Python executables
# -------------------------
install(
  PROGRAMS
    graspsam_ros2/graspsam_server.py
    graspsam_ros2/graspsam_cam2pose_server.py
  DESTINATION lib/${PROJECT_NAME}
)

# -------------------------
# Install Python package
# -------------------------
install(
  DIRECTORY graspsam_ros2
  DESTINATION lib/python3/dist-packages
)

# -------------------------
# Package marker
# -------------------------
ament_package()
