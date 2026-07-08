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
  -v "/media/csrobot/Data2/Datasets/Grasp-Anything:/media/Grasp-Anything:ro" \
  -v "/media/csrobot/Data2/Datasets/Jacquard_v2:/media/Jacquard_v2:ro" \
  -w /root/graspnet_ws/src/graspsam_ros2/graspsam_docker \
  ${IMAGE_NAME} #\
# bash -lc "[ -f setup_runtime.sh ] && bash setup_runtime.sh || true; exec bash"

# python eval.py   --dataset_name from_rgbd   --ckp_path trained_checkpoint/total_vit_t_default/jacquard/2026-02-28-03-03-42/epoch199.pth   --sam-encoder-type vit_t   --no-grasps 25 --use_crop 1 --remove_background 1
# python train.py --root ./datasets/Jacquard_Samples --save --sam-encoder-type vit_t