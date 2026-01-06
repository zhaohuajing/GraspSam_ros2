#!/usr/bin/env bash
set -e

IMAGE_NAME="graspsam:cu117"
CONTAINER_NAME="graspsam_dev"

HOST_WS="$HOME/graspnet_ws"
CONTAINER_WS="/root/graspnet_ws"

# Remove old container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[INFO] Removing existing container: ${CONTAINER_NAME}"
    docker rm -f ${CONTAINER_NAME}
fi

echo "[INFO] Starting container ${CONTAINER_NAME}"
docker run --rm -it \
    --name ${CONTAINER_NAME} \
    --gpus all \
    -v "${HOST_WS}:${CONTAINER_WS}" \
    -w "${CONTAINER_WS}/src/GraspSam_ros2/compare_GraspSAM" \
    ${IMAGE_NAME} \
    bash
