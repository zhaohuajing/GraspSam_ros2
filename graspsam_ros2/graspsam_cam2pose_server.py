#!/usr/bin/env python3
import os
import shlex
import json
import numpy as np
import subprocess
from pathlib import Path
from typing import List, Any, Dict, Optional
from geometry_msgs.msg import Pose, Point, PoseArray


import rclpy
from rclpy.node import Node

import tf2_ros
import tf_transformations as tft
import rclpy.duration

# IMPORTANT: package name should match your package.xml <name>...</name>
from graspsam_ros2.srv import RunGraspSAM
from graspsam_ros2.msg import Grasp


class GraspSAMCam2PoseServer(Node):
    """
    ROS2 service that runs GraspSAM eval.py inside a docker container via subprocess.

    Flow:
      1) Ensure a detached container is running
      2) docker exec into it and run eval.py
      3) Find latest output folder under compare_GraspSAM/grasp_outputs
      4) Parse grasps.json -> list[Grasp.msg]
      5) Return output_dir + grasps

      #> poses              geometry_msgs/Pose[]  # packed grasp pose in base frame
    """

    def __init__(self):
        super().__init__("graspsam_cam2pose_server")

        self.srv = self.create_service(RunGraspSAM, "/run_graspsam", self.handle_request)
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

        # Define frames and buffer 
        self.base_frame = 'simple_pedestal'   # or panda_link0 etc
        self.camera_frame = 'rgbd_camera/camera_link/rgbd_camera'  # must match TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


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

    def _docker_exec_eval(self, dataset_root: str, dataset_name: str, checkpoint_path: str, sam_encoder_type: str, no_grasps: int, seen_set: bool):
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
        dataset_arg = ""
        if dataset_name is not None and str(dataset_name).strip() != "":
            dataset_arg = f"--dataset_name {str(dataset_name).strip()}"

        seen_flag = " --seen-set" if bool(seen_set) else ""

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
            # f"python eval_edited_jac.py "
            # f"python eval_jac_from_git.py "
            f"--root {shlex.quote(dataset_root)} "
            # f"--dataset_name {shlex.quote(dataset_name)} "
            f"{dataset_arg} "
            f"--ckp_path {shlex.quote(checkpoint_path)} "
            f"--sam-encoder-type {shlex.quote(sam_encoder_type)} "
            f"--no-grasps {int(no_grasps)} "
            f"{seen_flag}"
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

            if isinstance(g, dict):
                # center convention: JSON should store x/y in image coordinates
                # Some older dumps might store "center":[y,x]
                if "center" in g and isinstance(g["center"], (list, tuple)) and len(g["center"]) >= 2:
                    # center is [y,x]
                    msg.x = float(g["center"][1])
                    msg.y = float(g["center"][0])
                else:
                    msg.x = float(g.get("x", 0.0))
                    msg.y = float(g.get("y", 0.0))

                msg.angle = float(g.get("angle", 0.0))

                # width in pixels: support both keys
                msg.width = float(g.get("width_px", g.get("width", 0.0)))

                # optional
                if hasattr(msg, "depth"):
                    msg.depth = float(g.get("depth", 0.0))

                # optional score/quality (only if field exists in msg)
                score = g.get("score", g.get("quality", 0.0))
                if hasattr(msg, "score"):
                    msg.score = float(score)
                if hasattr(msg, "quality"):
                    msg.quality = float(score)

                # 6D pose + metric width
                pos = g.get("pos", None)
                quat = g.get("quat", None)

                if pos is not None and quat is not None:
                    pos_arr = np.asarray(pos, dtype=np.float32).reshape(-1)
                    quat_arr = np.asarray(quat, dtype=np.float32).reshape(-1)

                    if len(pos_arr) >= 3 and len(quat_arr) >= 4:
                        # camera-frame pose from JSON
                        msg.pos_cam = [float(v) for v in pos_arr[:3]]
                        msg.quat_cam = [float(v) for v in quat_arr[:4]]

                # metric width (optional)
                if "width_m" in g:
                    msg.width_m = float(g.get("width_m", 0.0))

                # self.get_logger().info(
                #     f"msg.pos_cam= {msg.pos_cam}, msg.quat_cam = {msg.quat_cam}"
                # )

                ros_grasps.append(msg)

            elif isinstance(g, (list, tuple)) and len(g) >= 4:
                # Optional list format [x, y, angle, width, score]
                msg.x = float(g[0])
                msg.y = float(g[1])
                msg.angle = float(g[2])
                msg.width = float(g[3])

                if len(g) >= 5:
                    if hasattr(msg, "score"):
                        msg.score = float(g[4])
                    if hasattr(msg, "quality"):
                        msg.quality = float(g[4])

                ros_grasps.append(msg)

            else:
                # Unknown per-grasp format
                continue

        return ros_grasps

    # ------------------------------------------------------------------
    # Coordinate-frame helpers
    # ------------------------------------------------------------------
    def cgn_optical_to_ros_cam(self, T_cgn: np.ndarray) -> np.ndarray:
        """
        Contact-GraspNet grasps are expressed in the camera *optical* frame:
          x_right, y_down, z_forward.

        The URDF/TF frame `rgbd_camera/camera_link/rgbd_camera` is a standard
        ROS camera_link frame:
          X_forward, Y_left, Z_up.

        This applies the fixed rotation R_opt->cam_link so that the resulting
        4x4 matrix is in the ROS camera_link convention, rooted at the camera.
        """
        R = np.array([
            [0.0,  0.0, 1.0],   # z_opt -> X_cam
            [-1.0, 0.0, 0.0],   # x_opt -> -Y_cam
            [0.0, -1.0, 0.0],   # y_opt -> -Z_cam
        ], dtype=np.float64)

        # R = np.array([
        #     [1.0, 0.0, 0.0],   
        #     [0.0, 1.0, 0.0],   
        #     [0.0, 0.0, 1.0],  
        # ], dtype=np.float64)

        T_ros = np.eye(4, dtype=np.float64)
        T_ros[:3, :3] = R @ T_cgn[:3, :3]
        T_ros[:3, 3] = R @ T_cgn[:3, 3]
        return T_ros

    def transform_pose_array(self, pose_array: PoseArray,
                             from_frame: str,
                             to_frame: str) -> PoseArray:
        """
        Transform a PoseArray from `from_frame` to `to_frame` using TF2.
        Returns a new PoseArray in the target frame;
        if TF fails, returns the input pose_array.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                rclpy.time.Time(),  # latest
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().error(f"TF lookup {to_frame} <- {from_frame} failed: {e}")
            return pose_array

        trans = t.transform.translation
        rot = t.transform.rotation

        # 4x4 transform matrix base <- camera
        T_bc = tft.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        T_bc[0, 3] = trans.x
        T_bc[1, 3] = trans.y
        T_bc[2, 3] = trans.z

        out = PoseArray()
        out.header.frame_id = to_frame
        out.header.stamp = pose_array.header.stamp

        for p in pose_array.poses:
            # Pose in camera frame as 4x4
            q = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
            T_cp = tft.quaternion_matrix(q)
            T_cp[0, 3] = p.position.x
            T_cp[1, 3] = p.position.y
            T_cp[2, 3] = p.position.z

            # base <- camera <- pose
            T_bp = T_bc @ T_cp

            pos = T_bp[:3, 3]
            q_bp = tft.quaternion_from_matrix(T_bp)

            p_out = Pose()
            p_out.position.x = float(pos[0]) # - 0.45
            p_out.position.y = float(pos[1]) # + 0.05
            p_out.position.z = float(pos[2]) # - 0.7
            p_out.orientation.x = float(q_bp[0])
            p_out.orientation.y = float(q_bp[1])
            p_out.orientation.z = float(q_bp[2])
            p_out.orientation.w = float(q_bp[3])

            out.poses.append(p_out)

        return out


    # -----------------------------
    # ROS callback
    # -----------------------------
    def handle_request(self, request, response):
        try:
            self.get_logger().info("Received GraspSAM request.")
            self.get_logger().info(f"  dataset_root: {request.dataset_root}")
            self.get_logger().info(f"  dataset_name: {request.dataset_name}")
            self.get_logger().info(f"  checkpoint:   {request.checkpoint_path}")
            self.get_logger().info(f"  encoder:      {request.sam_encoder_type}")
            self.get_logger().info(f"  no_grasps:    {request.no_grasps}")
            self.get_logger().info(f"  seen_set:     {request.seen_set}")

            # 1) Ensure docker is running
            self._ensure_container()

            # 2) Run eval.py inside container
            proc = self._docker_exec_eval(
                dataset_root=request.dataset_root,
                dataset_name=request.dataset_name,
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

            response.message = (
                    f"eval.py finished, and grasps.json found in:\n{latest_dir}\n"
                )

            # 4) Parse JSON -> list[Grasp.msg]
            grasps_list = self.load_grasps_from_json(json_file)

            # Build PoseArray in camera frame from grasp messages
            grasps_cam_pa = PoseArray()
            grasps_cam_pa.header.stamp = self.get_clock().now().to_msg()
            grasps_cam_pa.header.frame_id = self.camera_frame

            valid_indices = []
            for i, g in enumerate(grasps_list):
                # only transform those that have cam pose filled
                if not hasattr(g, "pos_cam") or not hasattr(g, "quat_cam"):
                    continue
                if len(g.pos_cam) != 3 or len(g.quat_cam) != 4:
                    continue

                p = Pose()
                p.position.x = float(g.pos_cam[0])
                p.position.y = float(g.pos_cam[1])
                p.position.z = float(g.pos_cam[2])
                p.orientation.x = float(g.quat_cam[0])
                p.orientation.y = float(g.quat_cam[1])
                p.orientation.z = float(g.quat_cam[2])
                p.orientation.w = float(g.quat_cam[3])

                grasps_cam_pa.poses.append(p)
                valid_indices.append(i)

            # TF transform camera -> base
            grasps_base_pa = self.transform_pose_array(
                grasps_cam_pa,
                from_frame=self.camera_frame,
                to_frame=self.base_frame,
            )

            # Write base pose back into the corresponding Grasp msgs
            for pose_j, grasp_i in enumerate(valid_indices):
                pb = grasps_base_pa.poses[pose_j]
                g = grasps_list[grasp_i]
                g.pos_base = [float(pb.position.x), float(pb.position.y), float(pb.position.z)]
                g.quat_base = [
                    float(pb.orientation.x),
                    float(pb.orientation.y),
                    float(pb.orientation.z),
                    float(pb.orientation.w),
                ]

                g.pose_cam = grasps_cam_pa.poses[pose_j]
                g.pose_base = grasps_base_pa.poses[pose_j]

            # 5) Populate response
            response.success = True
            response.output_dir = str(latest_dir)
            response.message = (
                "GraspSAM eval completed.\n"
                f"Parsed grasps (total): {len(grasps_list)}\n"
                f"Transformed grasps: {len(valid_indices)}\n"
                f"Output dir: {latest_dir}\n"
                f"STDOUT (first 300 chars):\n{proc.stdout[:300]}\n"
                f"STDERR (first 300 chars):\n{proc.stderr[:300]}"
            )
            response.grasps = grasps_list


            self.get_logger().info(
                f"Responded with {len(valid_indices)} base-frame poses in '{self.base_frame}'."
            )
            return response


        except Exception as e:
            response.success = False
            response.message = str(e)
            response.output_dir = ""
            # try:
            #     response.grasps = []
            # except Exception:
            #     pass
            self.get_logger().error(f"GraspSAM request failed: {e}")
            return response


def main(args=None):
    rclpy.init(args=args)
    node = GraspSAMCam2PoseServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
