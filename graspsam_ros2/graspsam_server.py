#!/usr/bin/env python3
import os
import shlex
import json
import subprocess
from pathlib import Path
from typing import List, Any, Dict, Optional

import rclpy
from rclpy.node import Node

# IMPORTANT: package name should match your package.xml <name>...</name>
from graspsam_ros2.srv import RunGraspSAM
from graspsam_ros2.msg import Grasp


class GraspSAMServer(Node):
    """
    ROS2 service that runs GraspSAM eval.py inside a docker container via subprocess.

    Flow:
      1) Ensure a detached container is running
      2) docker exec into it and run eval.py
      3) Find latest output folder under compare_GraspSAM/grasp_outputs
      4) Parse grasps.json -> list[Grasp.msg]
      5) Return output_dir + grasps
    """

    def __init__(self):
        super().__init__("graspsam_server")

        self.srv = self.create_service(RunGraspSAM, "run_graspsam", self.handle_request)
        self.get_logger().info("GraspSAM server ready (runs eval.py inside Docker).")

        # Docker settings
        self.image_name = os.environ.get("GRASPSAM_IMAGE", "graspsam:cu117")
        self.container_name = os.environ.get("GRASPSAM_CONTAINER", "graspsam_dev")

        # Host workspace root (mounted into container)
        self.host_ws = os.environ.get("GRASPSAM_HOST_WS", str(Path.home() / "graspnet_ws"))
        self.container_ws = os.environ.get("GRASPSAM_CONTAINER_WS", "/root/graspnet_ws")

        # eval.py lives here inside container
        self.container_workdir = os.environ.get(
            "GRASPSAM_CONTAINER_WORKDIR",
            f"{self.container_ws}/src/graspsam_ros2/compare_GraspSAM",
        )

        # conda env inside container
        self.conda_env = os.environ.get("GRASPSAM_CONDA_ENV", "GraspSAM")

        # Host-side output root (where eval.py writes, visible on host via mount)
        self.host_output_root = Path(self.host_ws) / "src" / "graspsam_ros2" / "compare_GraspSAM" / "grasp_outputs"

    # -----------------------------
    # subprocess helpers
    # -----------------------------
    def _run(self, cmd_list, check=True):
        self.get_logger().debug(f"Running: {' '.join(shlex.quote(c) for c in cmd_list)}")
        p = subprocess.run(cmd_list, capture_output=True, text=True)
        if check and p.returncode != 0:
            raise RuntimeError(
                f"Command failed (rc={p.returncode}).\n"
                f"CMD: {' '.join(cmd_list)}\n"
                f"STDOUT:\n{p.stdout}\n"
                f"STDERR:\n{p.stderr}\n"
            )
        return p

    def _container_running(self) -> bool:
        p = self._run(["docker", "ps", "--format", "{{.Names}}"], check=True)
        names = {line.strip() for line in p.stdout.splitlines() if line.strip()}
        return self.container_name in names

    def _ensure_container(self):
        if self._container_running():
            return

        # If container exists but stopped, remove it
        p = self._run(["docker", "ps", "-a", "--format", "{{.Names}}"], check=True)
        existing = {line.strip() for line in p.stdout.splitlines() if line.strip()}
        if self.container_name in existing:
            self.get_logger().info(f"Removing existing container: {self.container_name}")
            self._run(["docker", "rm", "-f", self.container_name], check=True)

        self.get_logger().info(f"Starting container '{self.container_name}' from image '{self.image_name}'")

        # Detached container that stays alive; mount host workspace
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "--gpus", "all",
            # Optional shm help (safe to keep):
            "--ipc=host",
            "-v", f"{self.host_ws}:{self.container_ws}",
            "-w", self.container_workdir,
            self.image_name,
            "bash", "-lc", "sleep infinity"
        ]
        self._run(cmd, check=True)

    def _docker_exec_eval(self, dataset_root: str, checkpoint_path: str, sam_encoder_type: str, no_grasps: int, seen_set: bool):
        # conda_prefix = (
        #     "source /opt/conda/etc/profile.d/conda.sh && "
        #     f"conda activate {shlex.quote(self.conda_env)} && "
        # )

        conda_prefix = (
            "source /opt/conda/etc/profile.d/conda.sh && "
            f"conda activate {self.conda_env} && "
            "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && "
        )


        # Only include flag if your eval.py supports it
        seen_flag = "--seen-set" if seen_set else ""

        # cmd_str = (
        #     f"{conda_prefix}"
        #     f"python eval.py "
        #     f"--root {shlex.quote(dataset_root)} "
        #     f"--ckp_path {shlex.quote(checkpoint_path)} "
        #     f"--sam-encoder-type {shlex.quote(sam_encoder_type)} "
        #     f"--no-grasps {int(no_grasps)} "
        #     f"{seen_flag}"
        # ).strip()

        eval_dir = (
            f"{self.container_ws}/src/graspsam_ros2/compare_GraspSAM"
        )

        cmd_str = (
            f"{conda_prefix}"
            f"cd {shlex.quote(eval_dir)} && "
            f"python eval.py "
            f"--root {shlex.quote(dataset_root)} "
            f"--ckp_path {shlex.quote(checkpoint_path)} "
            f"--sam-encoder-type {shlex.quote(sam_encoder_type)} "
            f"--no-grasps {int(no_grasps)} "
        ).strip()


        cmd = ["docker", "exec", self.container_name, "bash", "-lc", cmd_str]
        return self._run(cmd, check=True)

    # -----------------------------
    # output parsing helpers
    # -----------------------------
    def _find_latest_output_dir(self) -> Optional[Path]:
        if not self.host_output_root.exists():
            return None
        subdirs = [p for p in self.host_output_root.iterdir() if p.is_dir()]
        if not subdirs:
            return None
        subdirs.sort(key=lambda p: p.stat().st_mtime)
        return subdirs[-1]

    def load_grasps_from_json(self, json_path: Path) -> List[Grasp]:
        """
        Load GraspSAM JSON output and convert to list[Grasp.msg].

        Supported JSON shapes:
          - {"grasps": [ ... ]}
          - [ ... ]  (list directly)
        Each grasp can be:
          - dict: {"x":..,"y":..,"angle":..,"width":..,"score":..,"depth":..}
          - list: [x, y, angle, width, score, (optional depth)]
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        ros_grasps = []

        # Normalize format
        if isinstance(data, dict) and "grasps" in data:
            grasps = data["grasps"]
        elif isinstance(data, list):
            grasps = data
        else:
            self.get_logger().warn(f"Unknown grasp JSON format: {type(data)}")
            return ros_grasps

        for g in grasps:
            msg = Grasp()

            # ---- Case 1: list format [x, y, angle, width, score]
            if isinstance(g, (list, tuple)) and len(g) >= 4:
                msg.x = float(g[0])
                msg.y = float(g[1])
                msg.angle = float(g[2])
                msg.width = float(g[3])
                msg.quality = float(g[4]) if len(g) > 4 else 0.0
                msg.depth = 0.0

            # ---- Case 2: dict with center
            elif isinstance(g, dict):
                if "center" in g:
                    msg.x = float(g["center"][0])
                    msg.y = float(g["center"][1])
                else:
                    msg.x = float(g.get("x", 0.0))
                    msg.y = float(g.get("y", 0.0))

                msg.angle = float(g.get("angle", 0.0))
                msg.width = float(g.get("width", 0.0))
                msg.quality = float(g.get("score", g.get("quality", 0.0)))
                msg.depth = float(g.get("depth", 0.0))

            else:
                continue

            ros_grasps.append(msg)

        return ros_grasps

    # -----------------------------
    # ROS callback
    # -----------------------------
    def handle_request(self, request, response):
        try:
            self.get_logger().info("Received GraspSAM request.")
            self.get_logger().info(f"  dataset_root: {request.dataset_root}")
            self.get_logger().info(f"  checkpoint:   {request.checkpoint_path}")
            self.get_logger().info(f"  encoder:      {request.sam_encoder_type}")
            self.get_logger().info(f"  no_grasps:    {request.no_grasps}")
            self.get_logger().info(f"  seen_set:     {request.seen_set}")

            # 1) Ensure docker is running
            self._ensure_container()

            # 2) Run eval.py inside container
            proc = self._docker_exec_eval(
                dataset_root=request.dataset_root,
                checkpoint_path=request.checkpoint_path,
                sam_encoder_type=request.sam_encoder_type,
                no_grasps=int(request.no_grasps),
                seen_set=bool(request.seen_set),
            )

            # 3) Locate latest output dir on HOST
            latest_dir = self._find_latest_output_dir()
            if latest_dir is None:
                response.success = False
                response.message = (
                    "eval.py finished, but no output directory was found under:\n"
                    f"{self.host_output_root}\n"
                    f"STDOUT (first 500 chars):\n{proc.stdout[:500]}\n"
                    f"STDERR (first 500 chars):\n{proc.stderr[:500]}"
                )
                response.output_dir = ""
                return response

            # json_file = latest_dir / "sample_0_grasps.json"

            json_files = sorted(latest_dir.glob("*_grasps.json"))
            if not json_files:
                raise RuntimeError("No *_grasps.json found")

            json_file = json_files[0]   # or loop over all


            if not json_file.exists():
                response.success = False
                response.message = (
                    f"eval.py finished, but grasps.json not found in:\n{latest_dir}\n"
                    f"STDOUT (first 500 chars):\n{proc.stdout[:500]}\n"
                    f"STDERR (first 500 chars):\n{proc.stderr[:500]}"
                )
                response.output_dir = str(latest_dir)
                return response

            # 4) Parse JSON -> ROS messages
            grasps_msg = self.load_grasps_from_json(json_file)

            # 5) Populate response
            response.success = True
            response.output_dir = str(latest_dir)
            response.message = (
                "GraspSAM eval completed.\n"
                f"Parsed grasps: {len(grasps_msg)}\n"
                f"Output dir: {latest_dir}\n"
                f"STDOUT (first 300 chars):\n{proc.stdout[:300]}\n"
                f"STDERR (first 300 chars):\n{proc.stderr[:300]}"
            )
            response.grasps = grasps_msg
            return response

        except Exception as e:
            response.success = False
            response.message = str(e)
            response.output_dir = ""
            try:
                response.grasps = []
            except Exception:
                pass
            self.get_logger().error(f"GraspSAM request failed: {e}")
            return response


def main(args=None):
    rclpy.init(args=args)
    node = GraspSAMServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
