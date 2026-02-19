# Grasp in image frame (2D planar grasp)

float32 x          # pixel x (center)
float32 y          # pixel y (center)
float32 angle      # radians, CCW
float32 width      # pixel width
float32 quality    # grasp score

# Optional depth at grasp center (meters), if available
float32 depth

# 6D grasp pose in camera frame (meters + quaternion)
# Convention: whatever your rectangle_to_pose_topdown() outputs (keep consistent).
float32[3] pos_cam
float32[4] quat_cam

# Same grasp pose transformed into base frame
float32[3] pos_base
float32[4] quat_base

# Metric gripper opening width (meters), if computed
float32 width_m

# packed grasp pose in base frame
geometry_msgs/Pose pose_cam
geometry_msgs/Pose pose_base

