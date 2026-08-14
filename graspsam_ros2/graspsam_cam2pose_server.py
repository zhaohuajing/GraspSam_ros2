#!/usr/bin/env python3
import os
import shlex
import json
import shutil
import time
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



'''
# Sample input command:

 - For Gazebo (Panda Sim):

ros2 run graspsam_ros2 graspsam_cam2pose_server --ros-args \
  -p base_frame:=simple_pedestal \
  -p camera_frame:=rgbd_camera/camera_link/rgbd_camera \
  -p use_optical_to_ros_cam:=true \
  -p custom_no_gt:=1 \
  -p sample_id:=0_from_rgbd \
  -p mask_id:=0 \
  -p remove_background:=0 \
  -p apply_mask_to_q:=1 \
  -p fx:=554.3827128226441 \
  -p fy:=554.3827128226441 \
  -p cx:=320.0 \
  -p cy:=240.0 \
  -p intr_w:=640 \
  -p intr_h:=480

- For Kinova Gen3 Real

ros2 run graspsam_ros2 graspsam_cam2pose_server --ros-args \
  -p base_frame:=camera_color_frame \
  -p camera_frame:=camera_link \
  -p use_optical_to_ros_cam:=true \
  -p custom_no_gt:=1 \
  -p sample_id:=0_from_rgbd \
  -p mask_id:=6 \
  -p remove_background:=0 \
  -p apply_mask_to_q:=1 \
  -p fx:=1297.673 \
  -p fy:=1298.631 \
  -p cx:=620.914 \
  -p cy:=238.280 \
  -p intr_w:=1280 \
  -p intr_h:=720


- Sample service call:

ros2 service call /run_graspsam graspsam_ros2/srv/RunGraspSAM "{
  dataset_root: './rgbd2jacquard/Kinova_Gen3_real_YCB/sample2_mnet_scene',
  dataset_name: 'jacquard',
  checkpoint_path: './trained_checkpoint/total_vit_t_default/jacquard/2026-02-28-04-40-49/epoch54.pth',
  sam_encoder_type: 'vit_t',
  no_grasps: 5,
  seen_set: false
}"

ros2 service call /run_graspsam graspsam_ros2/srv/RunGraspSAM "{
  dataset_root: './rgbd2jacquard/Kinova_Gen3_real_YCB/sample2_mnet_scene',
  dataset_name: 'jacquard',
  checkpoint_path: './trained_checkpoint/total_vit_t_default/jacquard/2026-02-28-04-40-49/epoch54.pth',
  sam_encoder_type: 'vit_t',
  no_grasps: 5,
  seen_set: false,
  custom_no_gt: true,
  sample_id: '0_from_rgbd',
  mask_id: 6,
  remove_background: false,
  apply_mask_to_q: true
}"

for which the server runs something like:

python eval.py \
  --root ./rgbd2jacquard/Kinova_Gen3_real_YCB/sample2_mnet_scene \
  --dataset_name jacquard \
  --ckp_path ./trained_checkpoint/total_vit_t_default/jacquard/2026-02-28-04-40-49/epoch54.pth \
  --sam-encoder-type vit_t \
  --no-grasps 5 \
  --custom_no_gt 1 \
  --sample-id 0_from_rgbd \
  --mask-id 6 \
  --remove_background 0 \
  --apply_mask_to_q 1 \
  --fx ... --fy ... --cx ... --cy ... --intr_w ... --intr_h ...

'''

def _env_bool(name: str, default: bool) -> bool:
    """Parse bool-like environment variables safely."""
    value = os.environ.get(name, None)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, None)
    if value is None or str(value).strip() == "":
        return int(default)
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, None)
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


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

        # Host-side output root (where eval.py writes, visible on host via mount).
        # These folders are created by the Docker container and may appear as
        # root-owned/locked on the host.  The server below copies each newest run
        # into compare_GraspSAM/results using the host ROS process, so the copied
        # files are directly editable from Ubuntu.
        self.host_output_root = Path(self.host_ws) / "src" / "graspsam_ros2" / "compare_GraspSAM" / "grasp_outputs"

        default_results_root = Path(self.host_ws) / "src" / "graspsam_ros2" / "compare_GraspSAM" / "results"
        default_backup_root = self.host_output_root
        self.declare_parameter("copy_outputs_to_results", _env_bool("GRASPSAM_COPY_OUTPUTS_TO_RESULTS", True))
        self.declare_parameter("save_results_backup", _env_bool("GRASPSAM_SAVE_RESULTS_BACKUP", True))
        self.declare_parameter("results_root", os.environ.get("GRASPSAM_RESULTS_ROOT", str(default_results_root)))
        self.declare_parameter("realtime_results_dir", os.environ.get("GRASPSAM_REALTIME_RESULTS_DIR", str(default_results_root / "realtime")))
        self.declare_parameter("backup_results_root", os.environ.get("GRASPSAM_BACKUP_RESULTS_ROOT", str(default_backup_root)))

        self.copy_outputs_to_results = bool(self.get_parameter("copy_outputs_to_results").value)
        self.save_results_backup = bool(self.get_parameter("save_results_backup").value)
        self.host_results_root = Path(str(self.get_parameter("results_root").value)).expanduser()
        self.realtime_results_dir = Path(str(self.get_parameter("realtime_results_dir").value)).expanduser()
        self.host_backup_results_root = Path(str(self.get_parameter("backup_results_root").value)).expanduser()

        # ------------------------------------------------------------------
        # ROS parameters for Gazebo vs Kinova Gen3
        #
        # For Gazebo Panda examples you likely used:
        #   base_frame   = simple_pedestal
        #   camera_frame = rgbd_camera/camera_link/rgbd_camera
        #
        # For Kinova Gen3, set these at launch time to the TF frames actually
        # present in your system, e.g.:
        #   base_frame   = base_link              (or your MoveIt planning frame)
        #   camera_frame = camera_link            (ROS camera_link-style frame)
        #
        # If camera_frame is an optical frame (x-right, y-down, z-forward),
        # set use_optical_to_ros_cam:=false because TF already knows the optical
        # frame convention. If camera_frame is camera_link-style, keep it true.
        # ------------------------------------------------------------------
        # self.declare_parameter("base_frame", os.environ.get("GRASPSAM_BASE_FRAME", "simple_pedestal"))
        # self.declare_parameter("camera_frame", os.environ.get("GRASPSAM_CAMERA_FRAME", "rgbd_camera/camera_link/rgbd_camera"))
        # self.declare_parameter("use_optical_to_ros_cam", _env_bool("GRASPSAM_USE_OPTICAL_TO_ROS_CAM", True))


        self.declare_parameter("base_frame", os.environ.get("GRASPSAM_BASE_FRAME", "base_link"))
        self.declare_parameter("camera_frame", os.environ.get("GRASPSAM_CAMERA_FRAME", "camera_color_frame")) # camera link resulted in OMPL grasp pose offset
        # self.declare_parameter("camera_frame", os.environ.get("GRASPSAM_CAMERA_FRAME", "camera_depth_frame")) # try using camera_depth_frame instead
        # self.declare_parameter("camera_frame", os.environ.get("GRASPSAM_CAMERA_FRAME", "camera_compensate_frame")) # try using camera_depth_frame instead; worked well before adjusting kortex version of TF for color-depth mapping, not after

        self.declare_parameter("use_optical_to_ros_cam", _env_bool("GRASPSAM_USE_OPTICAL_TO_ROS_CAM", False))

        # Optional fixed transform from the GraspSAM/CGN grasp frame to the
        # actual robot gripper frame.  These affect orientation only.
        self.declare_parameter('apply_gripper_frame_offset', False)
        self.declare_parameter('gripper_offset_rx_deg', 0.0)
        self.declare_parameter('gripper_offset_ry_deg', 0.0)
        self.declare_parameter('gripper_offset_rz_deg', 0.0)



        # Optional right-handed X/Y axis swap in the grasp/gripper frame.
        # A pure X<->Y swap is a reflection (det=-1), so this uses the same
        # right-handed 90-deg Z rotation convention as the earlier CGN-style
        # SceneReplica swap:
        #   x_new = y_old, y_new = -x_old, z_new = z_old
        #
        # Keep apply_scene_replica_xy_swap as a backward-compatible alias.
        self.declare_parameter('apply_gripper_frame_xy_swap', False)
        self.declare_parameter('apply_scene_replica_xy_swap', False)

        self.base_frame = self.get_parameter("base_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.use_optical_to_ros_cam = bool(self.get_parameter("use_optical_to_ros_cam").value)

        self.apply_gripper_frame_offset = bool(
            self.get_parameter('apply_gripper_frame_offset').value
        )
        self.gripper_offset_rx_deg = float(self.get_parameter('gripper_offset_rx_deg').value)
        self.gripper_offset_ry_deg = float(self.get_parameter('gripper_offset_ry_deg').value)
        self.gripper_offset_rz_deg = float(self.get_parameter('gripper_offset_rz_deg').value)

        self.apply_gripper_frame_xy_swap = bool(
            self.get_parameter('apply_gripper_frame_xy_swap').value
        )
        self.apply_scene_replica_xy_swap = bool(
            self.get_parameter('apply_scene_replica_xy_swap').value
        )


        # A temporay bias to manually adjust eef position in base frame to resolve potential camera-to-base TF offset, camera mounting/extrinsic error, or end_effector_link / Robotiq grasp-center TCP offset
        self.declare_parameter("base_y_offset", 0.022)
        self.base_y_offset = float(self.get_parameter("base_y_offset").value)

        # ------------------------------------------------------------------
        # eval.py options exposed as ROS parameters.
        # This lets the same server work with the custom Jacquard-like Kinova
        # RGB-D folders, without modifying RunGraspSAM.srv every time.
        # ------------------------------------------------------------------
        self.declare_parameter("eval_script", os.environ.get("GRASPSAM_EVAL_SCRIPT", "eval.py"))

        # Custom no-GT Jacquard-like input mode.
        # Default is enabled because your current Kinova/UOC pipeline uses
        # RGB/depth/mask input without meaningful *_grasps.txt annotations.
        self.declare_parameter("custom_no_gt", _env_int("GRASPSAM_CUSTOM_NO_GT", 1))
        self.declare_parameter("sample_id", os.environ.get("GRASPSAM_SAMPLE_ID", "0_from_rgbd"))
        self.declare_parameter("mask_id", _env_int("GRASPSAM_MASK_ID", 0))
        self.declare_parameter("remove_background", _env_int("GRASPSAM_REMOVE_BACKGROUND", 0))
        self.declare_parameter("apply_mask_to_q", _env_int("GRASPSAM_APPLY_MASK_TO_Q", 1))

        # Camera intrinsics corresponding to the RGB-D image before your
        # Jacquard-like pad-to-square/resize step. For Kinova, set these from
        # the CameraInfo of the exact image stream/cropped image used by UOC.
        # self.declare_parameter("fx", _env_float("GRASPSAM_FX", 554.3827128226441))
        # self.declare_parameter("fy", _env_float("GRASPSAM_FY", 554.3827128226441))
        # self.declare_parameter("cx", _env_float("GRASPSAM_CX", 320.0))
        # self.declare_parameter("cy", _env_float("GRASPSAM_CY", 240.0))
        # self.declare_parameter("intr_w", _env_int("GRASPSAM_INTR_W", 640))
        # self.declare_parameter("intr_h", _env_int("GRASPSAM_INTR_H", 480))

        self.declare_parameter("fx", _env_float("GRASPSAM_FX", 1297.673))
        self.declare_parameter("fy", _env_float("GRASPSAM_FY", 1298.631))
        self.declare_parameter("cx", _env_float("GRASPSAM_CX", 620.914))
        self.declare_parameter("cy", _env_float("GRASPSAM_CY", 238.280))
        self.declare_parameter("intr_w", _env_int("GRASPSAM_INTR_W", 1280))
        self.declare_parameter("intr_h", _env_int("GRASPSAM_INTR_H", 720))


        self.eval_script = self.get_parameter("eval_script").value
        self.custom_no_gt = int(self.get_parameter("custom_no_gt").value)
        self.sample_id = str(self.get_parameter("sample_id").value)
        self.mask_id = int(self.get_parameter("mask_id").value)
        self.remove_background = int(self.get_parameter("remove_background").value)
        self.apply_mask_to_q = int(self.get_parameter("apply_mask_to_q").value)

        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.cx = float(self.get_parameter("cx").value)
        self.cy = float(self.get_parameter("cy").value)
        self.intr_w = int(self.get_parameter("intr_w").value)
        self.intr_h = int(self.get_parameter("intr_h").value)

        self.get_logger().info(
            f"Frames: base_frame='{self.base_frame}', camera_frame='{self.camera_frame}', "
            f"use_optical_to_ros_cam={self.use_optical_to_ros_cam}"
        )
        self.get_logger().info(
            "eval.py options: "
            f"custom_no_gt={self.custom_no_gt}, sample_id='{self.sample_id}', mask_id={self.mask_id}, "
            f"remove_background={self.remove_background}, apply_mask_to_q={self.apply_mask_to_q}, "
            f"intrinsics=({self.fx}, {self.fy}, {self.cx}, {self.cy}), "
            f"intr_size=({self.intr_w}, {self.intr_h})"
        )
        self.get_logger().info(
            "Gripper-frame pose options: "
            f"apply_gripper_frame_offset={self.apply_gripper_frame_offset}, "
            f"rx/ry/rz_deg=({self.gripper_offset_rx_deg}, "
            f"{self.gripper_offset_ry_deg}, {self.gripper_offset_rz_deg}), "
            f"apply_gripper_frame_xy_swap={self.apply_gripper_frame_xy_swap}, "
            f"apply_scene_replica_xy_swap(alias)={self.apply_scene_replica_xy_swap}"
        )
        self.get_logger().info(
            "Result publishing options: "
            f"copy_outputs_to_results={self.copy_outputs_to_results}, "
            f"save_results_backup={self.save_results_backup}, "
            f"results_root='{self.host_results_root}', "
            f"realtime_results_dir='{self.realtime_results_dir}', "
            f"backup_results_root='{self.host_backup_results_root}'"
        )

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

    def _to_container_path(self, path_like: str) -> str:
        """
        Map a host workspace path to the corresponding path inside the GraspSAM
        Docker container. Relative paths and already-container paths are kept.

        Examples:
          /home/csrobot/graspnet_ws/src/... -> /root/graspnet_ws/src/...
          ./rgbd2jacquard/...              -> ./rgbd2jacquard/...
          /root/graspnet_ws/src/...        -> /root/graspnet_ws/src/...
        """
        if path_like is None:
            return ""

        p_str = os.path.expanduser(str(path_like).strip())
        if p_str == "":
            return p_str

        # Keep relative paths relative to container_workdir.
        if not os.path.isabs(p_str):
            return p_str

        # Already a container path.
        container_ws = os.path.normpath(self.container_ws)
        if os.path.normpath(p_str).startswith(container_ws):
            return p_str

        # Host workspace path -> container workspace path.
        host_ws = os.path.normpath(os.path.expanduser(self.host_ws))
        norm_p = os.path.normpath(p_str)
        if norm_p == host_ws or norm_p.startswith(host_ws + os.sep):
            rel = os.path.relpath(norm_p, host_ws)
            return os.path.join(container_ws, rel)

        # Otherwise leave unchanged; docker command will fail clearly if not mounted.
        return p_str


    def _docker_exec_eval(self, dataset_root: str, dataset_name: str, checkpoint_path: str,
                          sam_encoder_type: str, no_grasps: int, seen_set: bool,
                          custom_no_gt: bool, sample_id: str, mask_id: int,
                          remove_background: bool, apply_mask_to_q: bool):
        conda_prefix = (
            "source /opt/conda/etc/profile.d/conda.sh && "
            f"conda activate {self.conda_env} && "
            "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && "
        )

        # Map host paths from ROS request into container-visible paths.
        dataset_root_c = self._to_container_path(dataset_root)
        checkpoint_path_c = self._to_container_path(checkpoint_path)

        dataset_arg = ""
        if dataset_name is not None and str(dataset_name).strip() != "":
            dataset_arg = f"--dataset_name {shlex.quote(str(dataset_name).strip())}"

        seen_flag = " --seen-set" if bool(seen_set) else ""

        eval_dir = f"{self.container_ws}/src/graspsam_ros2/compare_GraspSAM"

        # Extra args for your current custom Kinova/UOC -> Jacquard-like workflow.
        # These are supported by the custom eval.py / JacquardDataset we prepared.
        # Per-request options. These intentionally use the service request values,
        # not the startup ROS parameters, so FlexBE / ros2 service call can choose
        # the target object mask at runtime.
        extra_eval_args = [
            f"--custom_no_gt {int(custom_no_gt)}",
            f"--sample-id {shlex.quote(str(sample_id))}",
            f"--mask-id {int(mask_id)}",
            f"--remove_background {int(remove_background)}",
            f"--apply_mask_to_q {int(apply_mask_to_q)}",
            f"--fx {float(self.fx)}",
            f"--fy {float(self.fy)}",
            f"--cx {float(self.cx)}",
            f"--cy {float(self.cy)}",
            f"--intr_w {int(self.intr_w)}",
            f"--intr_h {int(self.intr_h)}",
        ]
        extra_eval_arg_str = " ".join(extra_eval_args)

        cmd_str = (
            f"{conda_prefix}"
            f"cd {shlex.quote(eval_dir)} && "
            f"python {shlex.quote(str(self.eval_script))} "
            f"--root {shlex.quote(dataset_root_c)} "
            f"{dataset_arg} "
            f"--ckp_path {shlex.quote(checkpoint_path_c)} "
            f"--sam-encoder-type {shlex.quote(sam_encoder_type)} "
            f"--no-grasps {int(no_grasps)} "
            f"{extra_eval_arg_str} "
            f"{seen_flag}"
        ).strip()

        self.get_logger().info(f"Running GraspSAM in Docker with root: {dataset_root_c}")
        self.get_logger().info(f"Docker eval command:\n{cmd_str}")

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

    def _chmod_tree_user_rw(self, path: Path):
        """Make copied result folders convenient to browse/edit on the host.

        The copies are created by this ROS process, so ownership should already be
        the host user.  chmod is still applied to avoid preserving restrictive
        file modes from Docker-created outputs.
        """
        try:
            if path.is_dir():
                os.chmod(path, 0o775)
            elif path.exists():
                os.chmod(path, 0o664)
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    try:
                        os.chmod(Path(root) / d, 0o775)
                    except Exception:
                        pass
                for f in files:
                    try:
                        os.chmod(Path(root) / f, 0o664)
                    except Exception:
                        pass
        except Exception as e:
            self.get_logger().warn(f"Could not chmod result copy '{path}': {e}")

    def _copy_dir_contents_host_owned(self, src_dir: Path, dst_dir: Path):
        """Copy a Docker-created output directory into a host-created folder."""
        src_dir = Path(src_dir)
        dst_dir = Path(dst_dir)
        if not src_dir.is_dir():
            raise RuntimeError(f"Source output directory does not exist: {src_dir}")

        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        for child in src_dir.iterdir():
            target = dst_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target, symlinks=True)
            else:
                shutil.copy2(child, target)

        self._chmod_tree_user_rw(dst_dir)

    def _make_docker_output_host_accessible(self, host_path: Path):
        """Make a Docker-created bind-mounted output directory writable by the host user.

        eval.py creates grasp_outputs/run_* inside Docker, which can make the
        folders look locked on Ubuntu.  The container normally runs as root, so
        use docker exec to chown/chmod the bind-mounted directory back to the
        user running this ROS server.
        """
        host_path = Path(host_path)
        if not host_path.exists():
            return

        try:
            uid = os.getuid()
            gid = os.getgid()
            container_path = self._to_container_path(str(host_path))
            cmd = [
                "docker", "exec", self.container_name, "bash", "-lc",
                (
                    f"chown -R {uid}:{gid} {shlex.quote(container_path)} && "
                    f"chmod -R u+rwX,go+rX {shlex.quote(container_path)}"
                ),
            ]
            self._run(cmd, check=True)
        except Exception as e:
            self.get_logger().warn(
                f"Could not make Docker output host-accessible for '{host_path}': {e}"
            )

    def _publish_latest_output_to_results(self, latest_dir: Path):
        """Publish the newest output.

        - Keep timestamped run_* outputs under compare_GraspSAM/grasp_outputs.
        - Only copy the stable realtime folder into compare_GraspSAM/results/realtime.

        Returns:
            realtime_dir: stable folder for downstream FlexBE/plotter use
            backup_dir: timestamped run_* folder under grasp_outputs, or None
        """
        latest_dir = Path(latest_dir)

        backup_dir = None
        if self.save_results_backup:
            self.host_backup_results_root.mkdir(parents=True, exist_ok=True)
            # eval.py already writes run_* under grasp_outputs.  Make that folder
            # host-accessible instead of copying timestamped backups into results/.
            backup_dir = latest_dir
            self._make_docker_output_host_accessible(backup_dir)

        if not self.copy_outputs_to_results:
            return latest_dir, backup_dir

        self.host_results_root.mkdir(parents=True, exist_ok=True)

        realtime_dir = self.realtime_results_dir
        self._copy_dir_contents_host_owned(latest_dir, realtime_dir)

        # Keep results/ clean except for results/realtime.
        self._chmod_tree_user_rw(realtime_dir)
        return realtime_dir, backup_dir

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
                # NOTE: DO NOT REVERT X and Y order here!! Already converted in grasp_rectangle_to_pose util
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
            # T_cp[0, 3] = p.position.x
            # T_cp[1, 3] = p.position.y
            # T_cp[2, 3] = p.position.z

            T_cp[0, 3] = p.position.x # * 1.05 # temporary manual adjustment to resolve potentially camera intrinsics related kinova executionndisplacement
            T_cp[1, 3] = p.position.y # * 1.05 # temporary manual adjustment to resolve potentially camera intrinsics related kinova executionndisplacement
            T_cp[2, 3] = p.position.z

            # base <- camera <- pose
            T_bp = T_bc @ T_cp

            pos = T_bp[:3, 3]
            q_bp = tft.quaternion_from_matrix(T_bp)

            p_out = Pose()
            p_out.position.x = float(pos[0]) 
            p_out.position.y = float(pos[1]) + self.base_y_offset # MANUAL ADJUSTMENT IN BASE FRAME AFTER FIXED CAMERA INTRINSICS AND PADDING
            p_out.position.z = max(float(pos[2]), 0.15) # set min height as 0.15 (length from EEF frame to fingertip)
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
            self.get_logger().info(f"  custom_no_gt:      {request.custom_no_gt}")
            self.get_logger().info(f"  sample_id:         {request.sample_id}")
            self.get_logger().info(f"  mask_id:           {request.mask_id}")
            self.get_logger().info(f"  remove_background: {request.remove_background}")
            self.get_logger().info(f"  apply_mask_to_q:   {request.apply_mask_to_q}")

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
                custom_no_gt=bool(request.custom_no_gt),
                sample_id=str(request.sample_id),
                mask_id=int(request.mask_id),
                remove_background=bool(request.remove_background),
                apply_mask_to_q=bool(request.apply_mask_to_q),
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

            # Publish latest output: keep run_* under grasp_outputs, and copy only
            # the stable realtime folder into results/realtime for FlexBE/plotter.
            results_dir, backup_dir = self._publish_latest_output_to_results(latest_dir)

            # json_file = results_dir / "sample_0_grasps.json"

            json_files = sorted(results_dir.glob("*_grasps.json"))
            if not json_files:
                raise RuntimeError(f"No *_grasps.json found in published results dir: {results_dir}")

            json_file = json_files[0]   # or loop over all


            if not json_file.exists():
                response.success = False
                response.message = (
                    f"eval.py finished, but grasps.json not found in published results dir:\n{results_dir}\n"
                    f"Original Docker output dir:\n{latest_dir}\n"
                    f"STDOUT (first 500 chars):\n{proc.stdout[:500]}\n"
                    f"STDERR (first 500 chars):\n{proc.stderr[:500]}"
                )
                response.output_dir = str(results_dir)
                return response

            backup_msg = f"\nTimestamped run folder: {backup_dir}" if backup_dir is not None else ""
            response.message = (
                    f"eval.py finished, and grasps.json found in published results dir:\n{results_dir}\n"
                    f"Original Docker output dir: {latest_dir}"
                    f"{backup_msg}\n"
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

                # ----------------------------
                # 1) Build T_opt from JSON pose (optical frame)
                # ----------------------------
                q_opt = [float(g.quat_cam[0]), float(g.quat_cam[1]), float(g.quat_cam[2]), float(g.quat_cam[3])]
                T_opt = tft.quaternion_matrix(q_opt)
                T_opt[0, 3] = float(g.pos_cam[0])
                T_opt[1, 3] = float(g.pos_cam[1])
                T_opt[2, 3] = float(g.pos_cam[2])

                # ----------------------------
                # 2) Convert optical -> ROS camera_link convention if needed.
                #
                # JSON poses from eval.py are in camera optical coordinates.
                # If self.camera_frame is a ROS camera_link-style frame, keep
                # use_optical_to_ros_cam=True. If self.camera_frame is already
                # an optical TF frame, set use_optical_to_ros_cam=False.
                # ----------------------------
                if self.use_optical_to_ros_cam:
                    T_cam = self.cgn_optical_to_ros_cam(T_opt)
                else:
                    T_cam = T_opt


                # ----- Optional fixed grasp-frame -> robot-gripper-frame correction -----
                #
                # T_cam is camera_frame -> GraspSAM/CGN grasp frame.
                # Right-multiplying by T_gripper_offset gives:
                #   camera_frame -> robot gripper frame
                #
                # Translation is intentionally left unchanged; these parameters only
                # adjust the output orientation used by MoveIt.
                T_gripper_offset = np.eye(4, dtype=np.float64)

                if self.apply_gripper_frame_offset:
                    rx, ry, rz = np.deg2rad([
                        self.gripper_offset_rx_deg,
                        self.gripper_offset_ry_deg,
                        self.gripper_offset_rz_deg,
                    ])
                    T_gripper_offset = T_gripper_offset @ tft.euler_matrix(rx, ry, rz, 'sxyz')

                # Optional right-handed X/Y axis swap.  The legacy
                # apply_scene_replica_xy_swap parameter is kept as an alias for
                # compatibility with your earlier Contact-GraspNet server.
                if self.apply_gripper_frame_xy_swap or self.apply_scene_replica_xy_swap:
                    swap_xy = np.array([
                        [0.,  1., 0., 0.],
                        [-1., 0., 0., 0.],
                        [0.,  0., 1., 0.],
                        [0.,  0., 0., 1.],
                    ], dtype=np.float64)
                    T_gripper_offset = T_gripper_offset @ swap_xy

                if (self.apply_gripper_frame_offset or
                        self.apply_gripper_frame_xy_swap or
                        self.apply_scene_replica_xy_swap):
                    T_cam = T_cam @ T_gripper_offset
                # -----------------------------------------------------------------------


                # ----------------------------
                # 3) Store Pose in self.camera_frame
                # ----------------------------

                p = Pose()
                p.position.x = float(T_cam[0, 3])
                p.position.y = float(T_cam[1, 3])
                p.position.z = float(T_cam[2, 3])

                q_cam = tft.quaternion_from_matrix(T_cam)
                p.orientation.x = float(q_cam[0])
                p.orientation.y = float(q_cam[1])
                p.orientation.z = float(q_cam[2])
                p.orientation.w = float(q_cam[3])

                grasps_cam_pa.poses.append(p)
                valid_indices.append(i)

                # Optional: keep messages consistent with the converted camera_link pose
                g.pos_cam = [p.position.x, p.position.y, p.position.z]
                g.quat_cam = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
                g.pose_cam = p

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
            response.output_dir = str(results_dir)
            response.message = (
                "GraspSAM eval completed.\n"
                f"Parsed grasps (total): {len(grasps_list)}\n"
                f"Transformed grasps: {len(valid_indices)}\n"
                f"Output dir: {results_dir}\n"
                f"Original Docker output dir: {latest_dir}\n"
                f"Backup copy: {backup_dir}\n"
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
