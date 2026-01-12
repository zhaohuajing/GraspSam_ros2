#!/usr/bin/env python3

import os
import json
import subprocess
from pathlib import Path

import rclpy
from rclpy.node import Node

from GraspSam_ros2.srv import RunGraspSAM
from GraspSam_ros2.msg import Grasp


def load_grasps_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    ros_grasps = []
    grasps = data.get("grasps", data)

    for g in grasps:
        msg = Grasp()

        if isinstance(g, dict):
            msg.x = float(g["x"])
            msg.y = float(g["y"])
            msg.angle = float(g["angle"])
            msg.width = float(g["width"])
            msg.quality = float(g.get("score", g.get("quality", 0.0)))
            msg.depth = float(g.get("depth", 0.0))
        else:
            msg.x = float(g[0])
            msg.y = float(g[1])
            msg.angle = float(g[2])
            msg.width = float(g[3])
            msg.quality = float(g[4]) if len(g) > 4 else 0.0
            msg.depth = 0.0

        ros_grasps.append(msg)

    return ros_grasps


class GraspSamServer(Node):

    def __init__(self):
        super().__init__('graspsam_server')

        self.srv = self.create_service(
            RunGraspSAM,
            'run_graspsam',
            self.handle_request
        )

        self.get_logger().info("GraspSAM ROS2 server ready.")

    def handle_request(self, request, response):
        self.get_logger().info("Received GraspSAM request")

        # === paths ===
        workspace = Path(os.getcwd())
        docker_script = workspace / "GraspSam_docker" / "run_docker.sh"

        output_root = workspace / "compare_GraspSAM" / "grasp_outputs"
        output_root.mkdir(parents=True, exist_ok=True)

        # === command ===
        cmd = [
            str(docker_script),
            request.rgb_path,
            request.depth_path,
            request.mask_path,
            str(request.num_grasps),
            request.sam_encoder_type,
            request.checkpoint_path,
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            response.success = False
            response.message = f"GraspSAM failed: {e}"
            return response

        # === find latest output ===
        subdirs = sorted(output_root.glob("*"), key=os.path.getmtime)
        if not subdirs:
            response.success = False
            response.message = "No output directory produced."
            return response

        latest_dir = subdirs[-1]
        json_file = latest_dir / "grasps.json"

        if not json_file.exists():
            response.success = False
            response.message = f"No grasps.json found in {latest_dir}"
            return response

        # === parse grasps ===
        try:
            response.grasps = load_grasps_from_json(json_file)
        except Exception as e:
            response.success = False
            response.message = f"Failed to parse grasps: {e}"
            return response

        response.success = True
        response.message = str(latest_dir)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GraspSamServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
