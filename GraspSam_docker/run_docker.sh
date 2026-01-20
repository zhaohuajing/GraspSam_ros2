#!/usr/bin/env bash
set -e

IMAGE_NAME="graspsam:cu117"
CONTAINER_NAME="graspsam_dev"

# Kill and remove any existing container with same name
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "[INFO] Removing existing container: ${CONTAINER_NAME}"
  docker rm -f ${CONTAINER_NAME}
fi

# Run container
docker run --rm -it \
  --name ${CONTAINER_NAME} \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$HOME/graspnet_ws:/root/graspnet_ws" \
  -v "/media/csrobot/Data/Datasets/Grasp-Anything:/media/Grasp-Anything:ro" \
  -w /root/graspnet_ws/src/graspsam_ros2/compare_GraspSAM \
  ${IMAGE_NAME} \
  bash
