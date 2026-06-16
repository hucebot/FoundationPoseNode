"""
FoundationPose ROS2 node: subscribes to compressed RGB and depth images,
runs object detection (YOLO) and 6-DoF pose estimation (register + track).

Requires: ROS2 (rclpy), sensor_msgs, geometry_msgs, message_filters.
Run from workspace: python run_demo_ros2.py
  (or: ros2 run <your_pkg> run_demo_ros2.py if installed as a package)

Subscriptions:
  - /camera/color/image_raw/compressed (sensor_msgs/CompressedImage)
  - /camera/depth/image_raw/compressed (sensor_msgs/CompressedImage)
  - /camera/color/camera_info (sensor_msgs/CameraInfo) for intrinsics K
  - /orchestrator/pose/toggle_fp (std_msgs/Bool) to enable/disable the node
  - /orchestrator/pose/target_object (std_msgs/String) to set the target object class at runtime

Publishes:
  - object_pose (geometry_msgs/PoseStamped)
  - object_marker (visualization_msgs/Marker) mesh marker at pose
"""

import os
import time
import threading
from typing import Optional, Sequence

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker
from message_filters import Subscriber, ApproximateTimeSynchronizer

from foundationpose.estimater import *
from ultralytics import YOLO
from ultralytics.models.sam import SAM3SemanticPredictor

from scipy.spatial.transform import Rotation


def _quat_normalize_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_slerp_xyzw(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between quaternions in xyzw convention.
    Ensures shortest-path by flipping q1 if needed.
    """
    q0 = _quat_normalize_xyzw(q0)
    q1 = _quat_normalize_xyzw(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        # Nearly identical: fall back to lerp
        return _quat_normalize_xyzw(q0 + t * (q1 - q0))

    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * float(t)
    sin_theta = float(np.sin(theta))

    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(sin_theta / sin_theta_0)
    return _quat_normalize_xyzw((s0 * q0) + (s1 * q1))


class PoseFilter:
    """
    Position: Kalman filter (constant velocity), state [pos(3), vel(3)].
    Orientation: exponential smoothing via SLERP.
    """

    def __init__(self):
        self.initialized = False
        self.pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64)

    def reset(self):
        self.initialized = False

    def update(
        self,
        dt: float,
        meas_pos: np.ndarray,
        meas_quat_xyzw: np.ndarray,
        process_noise: float = 0.1,
        meas_noise: float = 0.05,
        slerp_factor: float = 0.15,
        max_dt_reinit: float = 1.0,
    ):
        meas_pos = np.asarray(meas_pos, dtype=np.float64).reshape(3)
        meas_quat_xyzw = _quat_normalize_xyzw(meas_quat_xyzw)
        dt = float(dt)

        if (not self.initialized) or (dt > max_dt_reinit):
            # (Re)initialize when starting, or when the time gap suggests tracking was lost.
            # We trust the measurement fully and reset velocity to 0.
            self.pos = meas_pos.copy()
            self.vel = np.zeros(3, dtype=np.float64)
            self.quat_xyzw = meas_quat_xyzw.copy()
            self.P = np.eye(6, dtype=np.float64)
            self.initialized = True
            return

        # --- Position KF (Constant Velocity) ---
        # State is x = [p, v]^T with p in meters and v in meters/second.
        # Motion model assumes constant velocity between frames:
        #   p_k = p_{k-1} + v_{k-1} * dt
        #   v_k = v_{k-1}
        # So the state transition is:
        #   x_k = F x_{k-1} + w,  with w ~ N(0, Q)
        F = np.eye(6, dtype=np.float64)
        F[:3, 3:] = np.eye(3, dtype=np.float64) * dt

        # Q: process noise covariance. 
        Q = np.eye(6, dtype=np.float64) * float(process_noise)
        x_pred = np.concatenate([self.pos, self.vel], axis=0).reshape(6, 1)
        x_pred = F @ x_pred
        P_pred = (F @ self.P @ F.T) + Q

        # H maps state -> measured components (extract p from [p, v]).
        H = np.zeros((3, 6), dtype=np.float64)
        H[:3, :3] = np.eye(3, dtype=np.float64)

        # R: measurement noise covariance.
        Rm = np.eye(3, dtype=np.float64) * float(meas_noise)
        S = (H @ P_pred @ H.T) + Rm
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Innovation/residual y = z - H x_pred, then correction x_upd = x_pred + K y.
        y = (meas_pos.reshape(3, 1) - (H @ x_pred))
        x_upd = x_pred + (K @ y)

        self.P = (np.eye(6, dtype=np.float64) - (K @ H)) @ P_pred
        self.pos = x_upd[:3, 0]
        self.vel = x_upd[3:, 0]

        # --- Orientation Smoothing (SLERP) ---
        # Orientation is not filtered with a full EKF here; we just smooth the measured quaternion.
        # slerp_factor in (0,1]: lower = more smoothing/lag, higher = more responsive.
        self.quat_xyzw = _quat_slerp_xyzw(self.quat_xyzw, meas_quat_xyzw, float(slerp_factor))



DET_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog',
    17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
    23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
    38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',
    43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush',
}

def decode_compressed_color(msg: CompressedImage) -> np.ndarray:
    """Decode CompressedImage to RGB (H, W, 3) uint8."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode color CompressedImage")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_compressed_depth(msg: CompressedImage, scale: float = 0.001) -> np.ndarray:
    # Skip the header (first 12 bytes)
    # https://github.com/ros-perception/image_transport_plugins
    depth_header_size = 12
    raw_data = msg.data[depth_header_size:]

    # Decode PNG (this is where uint16 is preserved)
    np_arr = np.frombuffer(raw_data, np.uint8)
    depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    if depth_img is None:
        raise RuntimeError("Failed to decode compressed depth image")

    if depth_img.dtype != np.uint16:
        raise RuntimeError(f"Expected uint16, got {depth_img.dtype}")

    return depth_img * scale

def symmetry_tfs_from_z_angles(z_angles):
    symmetry_tfs = []
    for z_angle in z_angles:
        # Pure Z rotation for Z-up meshes
        r = Rotation.from_euler("z", z_angle, degrees=True)
        tf = np.eye(4)
        tf[:3, :3] = r.as_matrix()
        symmetry_tfs.append(tf)
    return np.array(symmetry_tfs)


def _parse_angles_str(angles_str):
    """Parse a 'a1,a2,...' string of degrees into a list of floats, or None if empty."""
    if angles_str is None or angles_str == "":
        return None
    return [float(a) for a in angles_str.split(",")]


def symmetry_tfs_from_angles(x_angles_str=None, y_angles_str=None, z_angles_str=None):
    """
    Build the set of 4x4 symmetry transforms from per-axis angle strings (degrees).

    Each argument is a comma-separated string like "0,90,180,270" (or "" / None for no
    symmetry on that axis). The returned set is the Cartesian product of the per-axis
    rotations, composed as R = Rz @ Ry @ Rx (scipy "xyz" intrinsic order). When only
    z-angles are given this reproduces the previous pure-Z behavior.

    Returns None if no symmetry is requested on any axis.
    """
    xa = _parse_angles_str(x_angles_str)
    ya = _parse_angles_str(y_angles_str)
    za = _parse_angles_str(z_angles_str)
    if xa is None and ya is None and za is None:
        return None

    xs = xa if xa is not None else [0.0]
    ys = ya if ya is not None else [0.0]
    zs = za if za is not None else [0.0]

    symmetry_tfs = []
    for x_angle in xs:
        for y_angle in ys:
            for z_angle in zs:
                r = Rotation.from_euler("xyz", [x_angle, y_angle, z_angle], degrees=True)
                tf = np.eye(4)
                tf[:3, :3] = r.as_matrix()
                symmetry_tfs.append(tf)
    return np.array(symmetry_tfs)

def _orthonormal_basis_from_z(z_axis: np.ndarray) -> np.ndarray:
    """
    Right-handed rotation matrix [x, y, z] with given z column (object z in camera frame).

    Convention (camera frame): z-forward (depth), x-right, y-down (typical optical frame).
    - z column is the estimated object z-axis in camera coords.
    - x column is camera-forward (camera z) projected onto the plane orthogonal to object z.
    - y column completes the basis: y = z × x.
    """
    z = np.asarray(z_axis, dtype=np.float64).reshape(3)
    z /= np.linalg.norm(z) + 1e-12

    cam_fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # camera Z axis (depth)
    x_axis = cam_fwd - np.dot(cam_fwd, z) * z  # project forward into plane normal to z
    nx = np.linalg.norm(x_axis)
    if nx < 1e-8:
        # If object z is (near) parallel to camera forward, fall back to camera X then Y.
        cam_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = cam_x - np.dot(cam_x, z) * z
        nx = np.linalg.norm(x_axis)
        if nx < 1e-8:
            cam_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x_axis = cam_y - np.dot(cam_y, z) * z
            nx = np.linalg.norm(x_axis)

    x_axis /= nx + 1e-12
    y_axis = np.cross(z, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    return np.stack([x_axis, y_axis, z], axis=1)


def _orthonormal_basis_from_x(x_axis: np.ndarray) -> np.ndarray:
    """
    Right-handed rotation matrix [x, y, z] with given x column (object x in camera frame).

    Convention (camera frame): z-forward (depth), x-right, y-down (typical optical frame).
    - x column is the estimated object x-axis in camera coords.
    - z column is camera-forward (camera z) projected onto the plane orthogonal to object x.
    - y column completes the basis: y = z × x; then re-orthogonalize z = x × y.
    """
    x = np.asarray(x_axis, dtype=np.float64).reshape(3)
    x /= np.linalg.norm(x) + 1e-12

    cam_fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # camera Z axis (depth)
    z_axis = cam_fwd - np.dot(cam_fwd, x) * x  # project forward into plane normal to x
    nz = np.linalg.norm(z_axis)
    if nz < 1e-8:
        # If object x is (near) parallel to camera forward, fall back to camera X then Y.
        cam_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        z_axis = cam_x - np.dot(cam_x, x) * x
        nz = np.linalg.norm(z_axis)
        if nz < 1e-8:
            cam_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            z_axis = cam_y - np.dot(cam_y, x) * x
            nz = np.linalg.norm(z_axis)

    z_axis /= nz + 1e-12
    y_axis = np.cross(z_axis, x)
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    z_axis = np.cross(x, y_axis)
    z_axis /= np.linalg.norm(z_axis) + 1e-12
    return np.stack([x, y_axis, z_axis], axis=1)


def _orthonormal_basis_from_y(y_axis: np.ndarray) -> np.ndarray:
    """
    Right-handed rotation matrix [x, y, z] with given y column (object y in camera frame).

    Convention (camera frame): z-forward (depth), x-right, y-down (typical optical frame).
    - y column is the estimated object y-axis in camera coords.
    - z column is camera-forward (camera z) projected onto the plane orthogonal to object y.
    - x column completes the basis: x = y × z; then re-orthogonalize z = x × y.
    """
    y = np.asarray(y_axis, dtype=np.float64).reshape(3)
    y /= np.linalg.norm(y) + 1e-12

    cam_fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # camera Z axis (depth)
    z_axis = cam_fwd - np.dot(cam_fwd, y) * y  # project forward into plane normal to y
    nz = np.linalg.norm(z_axis)
    if nz < 1e-8:
        # If object y is (near) parallel to camera forward, fall back to camera X then Y.
        cam_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        z_axis = cam_x - np.dot(cam_x, y) * y
        nz = np.linalg.norm(z_axis)
        if nz < 1e-8:
            cam_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            z_axis = cam_y - np.dot(cam_y, y) * y
            nz = np.linalg.norm(z_axis)

    z_axis /= nz + 1e-12
    x_axis = np.cross(y, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    z_axis = np.cross(x_axis, y)
    z_axis /= np.linalg.norm(z_axis) + 1e-12
    return np.stack([x_axis, y, z_axis], axis=1)


def apply_rotate_z_in(R: np.ndarray, apply_z_in: Optional[Sequence[float]]) -> np.ndarray:
    """
    Keep the estimated object z-axis (column 2 of R); rebuild x,y in the plane orthogonal to z.

    apply_z_in:
      - None or (): leave R unchanged.
      - [0]: yaw around z fixed to 0 (x aligned with camera-forward as much as possible).
      - [0, P] with P in {90, 180, ...}: yaw angle (deg) reduced with np.mod(psi, P) into [0, P),
        e.g. P=90 → 115° → 25°; then R' = R_ref @ Rz(psi_reduced).
    """
    if apply_z_in is None:
        return np.asarray(R, dtype=np.float64).copy()
    rz = list(apply_z_in)
    if len(rz) == 0:
        return np.asarray(R, dtype=np.float64).copy()

    R = np.asarray(R, dtype=np.float64)
    z_axis = R[:, 2]
    R_ref = _orthonormal_basis_from_z(z_axis)
    x_ref, y_ref = R_ref[:, 0], R_ref[:, 1]

    if len(rz) == 1:
        if float(rz[0]) != 0.0:
            raise ValueError(f"apply_z_in with one element must be [0], got {apply_z_in!r}")
        return R_ref

    if len(rz) == 2:
        lo, hi = float(rz[0]), float(rz[1])
        if lo != 0.0:
            raise ValueError(f"apply_z_in two-element form must be [0, period], got {apply_z_in!r}")
        period = hi
        if period <= 0:
            raise ValueError(f"apply_z_in period must be > 0, got {period}")
        v = R[:, 0]
        psi_rad = np.arctan2(np.dot(v, y_ref), np.dot(v, x_ref))
        psi_deg = np.degrees(psi_rad)
        psi_rem = float(np.mod(psi_deg, period))
        Rz = Rotation.from_euler("z", psi_rem, degrees=True).as_matrix()
        return R_ref @ Rz

    raise ValueError(f"apply_z_in must be [0], [0, period], or empty/None; got {apply_z_in!r}")


def apply_rotate_x_in(R: np.ndarray, apply_x_in: Optional[Sequence[float]]) -> np.ndarray:
    """
    Keep the estimated object x-axis (column 0 of R); rebuild y,z in the plane orthogonal to x.

    apply_x_in:
      - None or (): leave R unchanged.
      - [0]: roll around x fixed to 0 (z aligned with camera-forward as much as possible).
      - [0, P] with P in {90, 180, ...}: roll angle (deg) reduced with np.mod(psi, P) into [0, P),
        then R' = R_ref @ Rx(psi_reduced).
    """
    if apply_x_in is None:
        return np.asarray(R, dtype=np.float64).copy()
    rx = list(apply_x_in)
    if len(rx) == 0:
        return np.asarray(R, dtype=np.float64).copy()

    R = np.asarray(R, dtype=np.float64)
    x_axis = R[:, 0]
    R_ref = _orthonormal_basis_from_x(x_axis)
    y_ref, z_ref = R_ref[:, 1], R_ref[:, 2]

    if len(rx) == 1:
        if float(rx[0]) != 0.0:
            raise ValueError(f"apply_x_in with one element must be [0], got {apply_x_in!r}")
        return R_ref

    if len(rx) == 2:
        lo, hi = float(rx[0]), float(rx[1])
        if lo != 0.0:
            raise ValueError(f"apply_x_in two-element form must be [0, period], got {apply_x_in!r}")
        period = hi
        if period <= 0:
            raise ValueError(f"apply_x_in period must be > 0, got {period}")
        v = R[:, 2]  # current z axis
        psi_rad = np.arctan2(np.dot(v, y_ref), np.dot(v, z_ref))
        psi_deg = np.degrees(psi_rad)
        psi_rem = float(np.mod(psi_deg, period))
        Rx = Rotation.from_euler("x", psi_rem, degrees=True).as_matrix()
        return R_ref @ Rx

    raise ValueError(f"apply_x_in must be [0], [0, period], or empty/None; got {apply_x_in!r}")


def apply_rotate_y_in(R: np.ndarray, apply_y_in: Optional[Sequence[float]]) -> np.ndarray:
    """
    Keep the estimated object y-axis (column 1 of R); rebuild x,z in the plane orthogonal to y.

    apply_y_in:
      - None or (): leave R unchanged.
      - [0]: pitch around y fixed to 0 (z aligned with camera-forward as much as possible).
      - [0, P] with P in {90, 180, ...}: pitch angle (deg) reduced with np.mod(psi, P) into [0, P),
        then R' = R_ref @ Ry(psi_reduced).
    """
    if apply_y_in is None:
        return np.asarray(R, dtype=np.float64).copy()
    ry = list(apply_y_in)
    if len(ry) == 0:
        return np.asarray(R, dtype=np.float64).copy()

    R = np.asarray(R, dtype=np.float64)
    y_axis = R[:, 1]
    R_ref = _orthonormal_basis_from_y(y_axis)
    x_ref, z_ref = R_ref[:, 0], R_ref[:, 2]

    if len(ry) == 1:
        if float(ry[0]) != 0.0:
            raise ValueError(f"apply_y_in with one element must be [0], got {apply_y_in!r}")
        return R_ref

    if len(ry) == 2:
        lo, hi = float(ry[0]), float(ry[1])
        if lo != 0.0:
            raise ValueError(f"apply_y_in two-element form must be [0, period], got {apply_y_in!r}")
        period = hi
        if period <= 0:
            raise ValueError(f"apply_y_in period must be > 0, got {period}")
        v = R[:, 2]  # current z axis
        psi_rad = np.arctan2(np.dot(v, x_ref), np.dot(v, z_ref))
        psi_deg = np.degrees(psi_rad)
        psi_rem = float(np.mod(psi_deg, period))
        Ry = Rotation.from_euler("y", psi_rem, degrees=True).as_matrix()
        return R_ref @ Ry

    raise ValueError(f"apply_y_in must be [0], [0, period], or empty/None; got {apply_y_in!r}")

# OBJECT_KEYS_TO_PARAMETERS = {
#     # "mustard": {"mesh_file": "./assets/hackathon2/mustard/mustard.obj", "symmetry_z_angles": "0,180", "target_object": "yellow bottle", "rotate_z_in":[0,180]},
#     # "juice": {"mesh_file": "./assets/hackathon2/juice/juice.obj", "symmetry_z_angles": "0,90,180,270", "target_object": "bottle", "rotate_z_in":[0,90]},
#     # "milk": {"mesh_file": "./assets/hackathon2/milk/milk.obj", "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330", "target_object": "white bottle", "rotate_z_in": [0]},
#     # "plate": {"mesh_file": "./assets/hackathon2/plate/plate.obj", "symmetry_z_angles": "0,90,180,270", "target_object": "red plate", "rotate_z_in": [0]},
#     # "gavottes": {"mesh_file": "./assets/hackathon2/gavottes/gavottes.obj", "symmetry_z_angles": "0,180", "target_object": "biscuit box", "rotate_z_in": [0,180]},
#     # "bowl": {"mesh_file": "./assets/hackathon2/bowl/bowl.obj", "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330", "target_object": "green bowl", "rotate_z_in": [0]},
# }

OBJECT_KEYS_TO_PARAMETERS = {
    # flips on x and y at 180 !
    "baguette" : {"mesh_file": "./assets/hackathon3/baguette/baguette.obj", 
                  "symmetry_x_angles": "0,180", 
                  "symmetry_y_angles": "0,180", 
                  "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330", 
                  "target_object": "bread", 
                  "constraint_yaw_in": 0,
                  "constraint_pitch_in": [0,180],
                  "constraint_roll_in": [0,180],
                  "apply_x_in": None,
                  "apply_y_in": None,
                  "apply_z_in": None},
    
    # seems okay and does not flip
    "banana" : {"mesh_file": "./assets/hackathon3/banana/banana.obj", 
                "symmetry_x_angles": "", 
                "symmetry_y_angles": "", 
                "symmetry_z_angles": "", 
                "target_object": "banana",
                "constraint_yaw_in": None,
                "constraint_pitch_in": None,
                "constraint_roll_in": None,
                "apply_x_in": None,
                "apply_y_in": None,
                "apply_z_in": None},
    
    # still flips up and down, adjusted size to retry
    "coffeecan" : {"mesh_file": "./assets/hackathon3/coffeecan/coffeecan.obj", 
                   "symmetry_x_angles": "", 
                   "symmetry_y_angles": "", 
                   "symmetry_z_angles": "", 
                   "target_object": "blue container", 
                   "constraint_yaw_in": 0,
                   "constraint_pitch_in": None,
                   "constraint_roll_in": None,
                   "apply_x_in": None,
                   "apply_y_in": None,
                   "apply_z_in": None},
    
    # lots of flipping difficult to stabilize
    "egg" : {"mesh_file": "./assets/hackathon3/egg/egg.obj", 
             "symmetry_x_angles": "0,180", 
             "symmetry_y_angles": "0,180", 
             "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330", 
             "target_object": "egg", 
             "constraint_yaw_in": 0,
             "constraint_pitch_in": 0,
             "constraint_roll_in": 0,
             "apply_x_in": None,
             "apply_y_in": None,
             "apply_z_in": None},
    
    # good it seems that does not flip up/down but the handle does not always get oriented correctly
    "flowercup" : {"mesh_file": "./assets/hackathon3/flowercup/flowercup.obj", 
                   "symmetry_x_angles": "", 
                   "symmetry_y_angles": "", 
                   "symmetry_z_angles": "", 
                   "target_object": "yellow mug", 
                   "constraint_yaw_in": None,
                   "constraint_pitch_in": None,
                   "constraint_roll_in": None,
                   "apply_x_in": None,
                   "apply_y_in": None,
                   "apply_z_in": None},
    
    # not detected with keyword jam
    "jam" : {"mesh_file": "./assets/hackathon3/jam/jam.obj", 
             "symmetry_x_angles": "", 
             "symmetry_y_angles": "", 
             "symmetry_z_angles": "", 
             "target_object": "orange jam", 
             "constraint_yaw_in": None,
             "constraint_pitch_in": None,
             "constraint_roll_in": None,
             "apply_x_in": None,
             "apply_y_in": None,
             "apply_z_in": None},
    
    # ok
    "milk" : {"mesh_file": "./assets/hackathon3/milk/milk.obj", 
              "symmetry_x_angles": "", 
              "symmetry_y_angles": "", 
              "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330", 
              "target_object": "white bottle", 
              "constraint_yaw_in": 0,
              "constraint_pitch_in": None,
              "constraint_roll_in": None,
              "apply_x_in": None,
              "apply_y_in": None,
              "apply_z_in": None},
    
    # hard to find but seems good when found... investigate, maybe a bigger one !
    "minicheese" : {"mesh_file": "./assets/hackathon3/minicheese/minicheese.obj", 
                    "symmetry_x_angles": "", 
                    "symmetry_y_angles": "", 
                    "symmetry_z_angles": "", 
                    "target_object": "triangle cheese", 
                    "constraint_yaw_in": None,
                    "constraint_pitch_in": None,
                    "constraint_roll_in": None,
                    "apply_x_in": None,
                    "apply_y_in": None,
                    "apply_z_in": None},
    
    # seems robust could be a good anchor point 
    "pan" : {"mesh_file": "./assets/hackathon3/pan/pan.obj", 
             "symmetry_x_angles": "", 
             "symmetry_y_angles": "", 
             "symmetry_z_angles": "", 
             "target_object": "pan", 
             "constraint_yaw_in": None,
             "constraint_pitch_in": None,
             "constraint_roll_in": None,
             "apply_x_in": None,
             "apply_y_in": None,
             "apply_z_in": None},
    
    # problem gets a lot of multiple objects + rotations
    "redapple" : {"mesh_file": "./assets/hackathon3/redapple/redapple.obj", 
                  "symmetry_x_angles": "0,180", 
                  "symmetry_y_angles": "0,180", 
                  "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330", 
                  "target_object": "red apple", 
                  "constraint_yaw_in": 0,
                  "constraint_pitch_in": 0,
                  "constraint_roll_in": 0,
                  "apply_x_in": None,
                  "apply_y_in": None,
                  "apply_z_in": None},
    
    "smallmilk" : {"mesh_file": "./assets/hackathon3/smallmilk/smallmilk.obj", 
                   "symmetry_x_angles": "", 
                   "symmetry_y_angles": "", 
                   "symmetry_z_angles": "", 
                   "target_object": "white bottle", 
                   "constraint_yaw_in": 0,
                   "constraint_pitch_in": None,
                   "constraint_roll_in": None,
                   "apply_x_in": None,
                   "apply_y_in": None,
                   "apply_z_in": None},
    
    "smallsanpellegrino" : {"mesh_file": "./assets/hackathon3/smallsanpellegrino/smallsanpellegrino.obj", 
                            "symmetry_x_angles": "", 
                            "symmetry_y_angles": "", 
                            "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330", 
                            "target_object": "green bottle", 
                            "constraint_yaw_in": 0,
                            "constraint_pitch_in": None,
                            "constraint_roll_in": None,
                            "apply_x_in": None,
                            "apply_y_in": None,
                            "apply_z_in": None},
    
    # hard to detect and flips on several axes
    "spam" : {"mesh_file": "./assets/hackathon3/spam/spam.obj", 
              "symmetry_x_angles": "", 
              "symmetry_y_angles": "", 
              "symmetry_z_angles": "", 
              "target_object": "blue container", 
              "constraint_yaw_in": [0,180],
              "constraint_pitch_in": [0,180],
              "constraint_roll_in": [0,180],
              "apply_x_in": None,
              "apply_y_in": None,
              "apply_z_in": None},
    
    # okay but flips on one axis
    "ycbmustard" : {"mesh_file": "./assets/hackathon3/ycbmustard/ycbmustard.obj", 
                    "symmetry_x_angles": "", 
                    "symmetry_y_angles": "", 
                    "symmetry_z_angles": "", 
                    "target_object": "yellow bottle", 
                    "constraint_yaw_in": [0,180],
                    "constraint_pitch_in": None,
                    "constraint_roll_in": None,
                    "apply_x_in": None,
                    "apply_y_in": None,
                    "apply_z_in": None},
}

class FoundationPoseROS2Node(Node):
    def __init__(self, args):
        super().__init__("foundation_pose_node")

        # Declare ROS parameters
        self.declare_parameter("mesh_file", args.mesh_file)
        self.declare_parameter("target_object", args.target_object)
        self.declare_parameter("est_refine_iter", args.est_refine_iter)
        self.declare_parameter("track_refine_iter", args.track_refine_iter)
        self.declare_parameter("debug", args.debug)
        self.declare_parameter("debug_dir", args.debug_dir)
        self.declare_parameter("depth_scale", args.depth_scale)
        self.declare_parameter("color_topic", args.color_topic)
        self.declare_parameter("depth_topic", args.depth_topic)
        self.declare_parameter("camera_info_topic", args.camera_info_topic)
        self.declare_parameter("pose_frame_id", args.pose_frame_id)
        self.declare_parameter("slop", args.slop)
        self.declare_parameter("marker_mesh_scale", 1.0)
        self.declare_parameter("marker_mesh_use_embedded_materials", True)
        self.declare_parameter("pose_filter_process_noise", 0.1)
        self.declare_parameter("pose_filter_meas_noise", 0.05)
        self.declare_parameter("pose_filter_slerp_factor", 0.15)
        self.declare_parameter("pose_filter_reset_lost_frames", 15)

        # Set current code directory
        code_dir = os.path.dirname(os.path.realpath(__file__))

        # Get parameters
        self.mesh_file = self.get_parameter("mesh_file").value
        assert(os.path.exists(self.mesh_file)), f"Mesh file {self.mesh_file} does not exist"
        mesh_file_basename = os.path.basename(self.mesh_file)
        mesh_file_rn = mesh_file_basename.split(".")[0]
        self._marker_mesh_resource = f"file:///mesh_assets/{mesh_file_rn}/{mesh_file_basename}"

        _abs_mesh = os.path.normpath(os.path.abspath(self.mesh_file))
        self.apply_z_in: Optional[Sequence[float]] = None
        self.apply_x_in: Optional[Sequence[float]] = None
        self.apply_y_in: Optional[Sequence[float]] = None
        self.constraint_yaw_in = None
        self.constraint_pitch_in = None
        self.constraint_roll_in = None
        for params in OBJECT_KEYS_TO_PARAMETERS.values():
            if mesh_file_basename in params["mesh_file"]:
                self.apply_z_in = params.get("apply_z_in")
                self.apply_x_in = params.get("apply_x_in")
                self.apply_y_in = params.get("apply_y_in")
                self.constraint_yaw_in = params.get("constraint_yaw_in")
                self.constraint_pitch_in = params.get("constraint_pitch_in")
                self.constraint_roll_in = params.get("constraint_roll_in")
                print(f"Apply z in: {self.apply_z_in}")
                break
        

        # Get debug directory and create if it doesn't exist
        self.debug_dir = self.get_parameter("debug_dir").value
        if not self.debug_dir:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.debug_dir = f"{code_dir}/debug_node/{timestamp}_{args.target_object}"
            
            if args.debug > 0:
                os.makedirs(self.debug_dir, exist_ok=True)

        # Set parameters
        self.target_object = self.get_parameter("target_object").value
        self.est_refine_iter = self.get_parameter("est_refine_iter").value
        self.track_refine_iter = self.get_parameter("track_refine_iter").value
        self.debug = self.get_parameter("debug").value
        self.depth_scale = self.get_parameter("depth_scale").value
        self.pose_frame_id = self.get_parameter("pose_frame_id").value
        self.slop = self.get_parameter("slop").value
        self.seg_model_name = args.seg_model_name
        self.resize_factor = args.resize_factor
        self.min_initial_detection_counter = args.min_initial_detection_counter
        self.enable_pose_tracking = args.enable_pose_tracking
        self.seg_model_type = args.seg_model_type
        self.symmetry_x_angles = args.symmetry_x_angles
        self.symmetry_y_angles = args.symmetry_y_angles
        self.symmetry_z_angles = args.symmetry_z_angles
        self.fp_verbosity = args.fp_verbosity
        self.use_kalman_filter = args.use_kalman_filter
        if self.fp_verbosity not in ["debug", "info", "warning", "error", "critical"]:
            raise ValueError(f"Invalid verbosity: {self.fp_verbosity}. Valid: debug, info, warning, error, critical")
        
        # Make some checks on the parameters
        assert(self.seg_model_type in ["sam3", "yolo"]), f"Invalid segmentation model type: {self.seg_model_type}"
        if self.seg_model_type == "sam3":
            self.seg_model_name = "sam3.pt"
        elif self.seg_model_type == "yolo":
            assert("yolo" in self.seg_model_name), f"Invalid YOLO model name: {self.seg_model_name}"

        coco_names = list(DET_NAMES.values())
        if self.seg_model_type == "yolo":
            assert(self.target_object in coco_names), f"Invalid target object: {self.target_object} (must be one of {coco_names})"
        
        # Print parameters
        self.get_logger().debug("==== PARAMETERS ====")
        self.get_logger().debug(f"Mesh file: {self.mesh_file}")
        self.get_logger().debug(f"Target object: {self.target_object}")
        self.get_logger().debug(f"Est refine iter: {self.est_refine_iter}")
        self.get_logger().debug(f"Track refine iter: {self.track_refine_iter}")
        self.get_logger().debug(f"Debug: {self.debug}")
        self.get_logger().debug(f"Debug dir: {self.debug_dir}")
        self.get_logger().debug(f"Depth scale: {self.depth_scale}")
        self.get_logger().debug(f"Pose frame id: {self.pose_frame_id}")
        self.get_logger().debug(f"Slop: {self.slop}")
        self.get_logger().debug(f"Resize factor: {self.resize_factor}")
        self.get_logger().debug(f"Min initial detection counter: {self.min_initial_detection_counter}")
        self.get_logger().debug(f"Enable pose tracking: {self.enable_pose_tracking}")
        self.get_logger().debug(f"Symmetry x angles: {self.symmetry_x_angles}")
        self.get_logger().debug(f"Symmetry y angles: {self.symmetry_y_angles}")
        self.get_logger().debug(f"Symmetry z angles: {self.symmetry_z_angles}")
        self.get_logger().debug(f"Apply z in: {self.apply_z_in}")
        self.get_logger().debug(f"Apply x in: {self.apply_x_in}")
        self.get_logger().debug(f"Apply y in: {self.apply_y_in}")
        self.get_logger().debug(f"Constraint yaw in: {self.constraint_yaw_in}")
        self.get_logger().debug(f"Constraint pitch in: {self.constraint_pitch_in}")
        self.get_logger().debug(f"Constraint roll in: {self.constraint_roll_in}")
        self.get_logger().debug(f"Use Kalman filter: {self.use_kalman_filter}")
        
        self.K = None # to be set by camera info callback
        self.est = None # to be set by estimator initialization
        self.current_phase = "NotInitialized"
        self.pose_last = None
        self.to_origin = None
        self.bbox = None
        self.frame_count = 0
        self._lock = threading.Lock()
        self._processing = False
        
        self.is_on = True #False
        self._prev_is_on = self.is_on
        
        self.initial_detection_counter = 0
        
        self.rgbd_frames_counter_received = 0
        self.rgbd_frames_counter_processed = 0
        
        # Set logger and seed (for estimater)
        verbosity = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR, "critical": logging.CRITICAL}
        set_logging_format(level=verbosity[self.fp_verbosity])
        set_seed(0)

        # Load mesh and compute bounds
        mesh = trimesh.load(self.mesh_file, force="mesh", skip_materials=True)
        self.to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        self.get_logger().info(f"Mesh lodaed from {self.mesh_file} | Bounds: {self.bbox.flatten()}")
        
        # Initialize segmentation / detection model
        self.get_logger().info(f"Initializing segmentation model {self.seg_model_type} ({self.seg_model_name})...")
        if self.seg_model_type == "sam3":
            # Initialize predictor with configuration
            overrides = dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=f"sam3/{self.seg_model_name}",
                half=True,  # Use FP16 for faster inference
                save=False,
            )
            self.seg_model = SAM3SemanticPredictor(overrides=overrides)
            # run a fake pass to warm up the model
            self.seg_model.set_image(np.zeros((1080, 1920, 3), dtype=np.uint8))
            self.seg_model(text=[self.target_object])
        elif self.seg_model_type == "yolo":
            self.seg_model = YOLO(self.seg_model_name)
        else:
            raise ValueError(f"Invalid segmentation model type: {self.seg_model_type}")
        self.get_logger().info(f"Segmentation model {self.seg_model_type} ({self.seg_model_name}) initialized")
        
        # Load symmetry transforms (combined over x, y, z axes)
        symmetry_tfs = symmetry_tfs_from_angles(
            self.symmetry_x_angles, self.symmetry_y_angles, self.symmetry_z_angles
        )
        if symmetry_tfs is not None:
            self.get_logger().debug(f"Symmetry transforms: {symmetry_tfs.shape}")
        else:
            self.get_logger().debug(f"No symmetry transforms")
        
        # Initialize estimator
        self.get_logger().info("Initializing estimator...")
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=self.debug_dir,
            debug=self.debug,
            glctx=glctx,
            symmetry_tfs=symmetry_tfs,
            meshname=mesh_file_rn,
        )
        self.get_logger().info("FoundationPose estimator initialized")
        
        # Update current phase
        self.current_phase = "Initialized"
        
        # Initialize ROS2 subscribers and publishers
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=15, # allow for 15 frames to be buffered
        )
        qos_info = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_marker = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._camera_info_cb,
            qos_info,
        )

        self._pose_pub = self.create_publisher(
            PoseStamped,
            "object_pose",
            1,
        )
        self._marker_pub = self.create_publisher(
            Marker,
            "object_marker",
            qos_marker,
        )

        self._toggle_fp_sub = self.create_subscription(
            Bool,
            "/orchestrator/pose/toggle_fp",
            self._toggle_fp_cb,
            1,
        )

        self._target_object_sub = self.create_subscription(
            String,
            "/orchestrator/pose/target_object",
            self._target_object_cb,
            1,
        )

        sub_color = Subscriber(
            self,
            CompressedImage,
            self.get_parameter("color_topic").value,
            qos_profile=qos_sensor,
        )
        sub_depth = Subscriber(
            self,
            CompressedImage,
            self.get_parameter("depth_topic").value,
            qos_profile=qos_sensor,
        )
        self._sync = ApproximateTimeSynchronizer(
            [sub_color, sub_depth],
            queue_size=100,
            slop=self.slop,
        )
        self._sync.registerCallback(self._rgbd_cb)
        
        self.get_logger().info(
            f"Subscribed to {self.get_parameter('color_topic').value} and {self.get_parameter('depth_topic').value}; waiting for camera_info and RGBD messages"
        )
        
        self.current_phase = "WaitingForCameraInfo"
        
        self.get_logger().info("FoundationPose ROS2 node initialized")
        
        # Optional pose filter (position KF + orientation SLERP smoothing)
        self._pose_filter = PoseFilter()
        self._pose_filter_last_stamp_s: Optional[float] = None
        self._pose_filter_lost_frames = 0

    def _reset_pose_filter(self, reason: str):
        self._pose_filter.reset()
        self._pose_filter_last_stamp_s = None
        self._pose_filter_lost_frames = 0
        self.get_logger().info(f"Pose filter reset: {reason}")

    def _note_pose_lost_for_filter(self):
        if not self.use_kalman_filter:
            return
        self._pose_filter_lost_frames += 1
        lost_thr = int(self.get_parameter("pose_filter_reset_lost_frames").value)
        if self._pose_filter_lost_frames > lost_thr:
            self._reset_pose_filter(f"object lost for > {lost_thr} frames")

    def _toggle_fp_cb(self, msg: Bool):
        prev = self.is_on
        self.is_on = msg.data
        self._prev_is_on = prev
        self.get_logger().info(f"FoundationPose toggled: is_on = {self.is_on}")
        
        if (not prev) and self.is_on:
            self._reset_pose_filter("node toggled off->on")
        
        if msg.data == False:
            if self.current_phase == "PoseTracking" or self.current_phase == "StartPoseTracking":
                self.get_logger().info("Stopping pose tracking back to detecting for later")
                self.current_phase = "DetectingAgain"

    def _target_object_cb(self, msg: String):
        new_target = msg.data.strip()
        if not new_target:
            self.get_logger().error("Received empty target object, ignoring")
            return
        
        if new_target.startswith("mesh_update_"):
            # it will be a full update with new mesh
            self.get_logger().info(f"Received mesh update request: {new_target}. Will restart the estimator with new mesh")
            self._lock.acquire()
            key_name = new_target.replace("mesh_update_", "") # name of target e.g.
            
            if key_name not in OBJECT_KEYS_TO_PARAMETERS:
                self.get_logger().error(f"Invalid key name: {key_name}. Valid: {list(OBJECT_KEYS_TO_PARAMETERS.keys())}")
                self._lock.release()
                return
            
            self.mesh_file = OBJECT_KEYS_TO_PARAMETERS[key_name]["mesh_file"]
            if not os.path.exists(self.mesh_file):
                self.get_logger().error(f"Mesh file {self.mesh_file} does not exist")
                self._lock.release()
                return
            
            basename = os.path.basename(self.mesh_file)
            rn = basename.split(".")[0]
            self._marker_mesh_resource = f"file:///mesh_assets/{rn}/{basename}"
            
            self.symmetry_x_angles = OBJECT_KEYS_TO_PARAMETERS[key_name].get("symmetry_x_angles", "")
            self.symmetry_y_angles = OBJECT_KEYS_TO_PARAMETERS[key_name].get("symmetry_y_angles", "")
            self.symmetry_z_angles = OBJECT_KEYS_TO_PARAMETERS[key_name]["symmetry_z_angles"]
            self.target_object = OBJECT_KEYS_TO_PARAMETERS[key_name]["target_object"]
            self.apply_z_in = OBJECT_KEYS_TO_PARAMETERS[key_name].get("apply_z_in")
            self.apply_x_in = OBJECT_KEYS_TO_PARAMETERS[key_name].get("apply_x_in")
            self.apply_y_in = OBJECT_KEYS_TO_PARAMETERS[key_name].get("apply_y_in")
            self.constraint_yaw_in = OBJECT_KEYS_TO_PARAMETERS[key_name].get("constraint_yaw_in")
            self.constraint_pitch_in = OBJECT_KEYS_TO_PARAMETERS[key_name].get("constraint_pitch_in")
            self.constraint_roll_in = OBJECT_KEYS_TO_PARAMETERS[key_name].get("constraint_roll_in")
            
            del self.est.scorer # delete the old score predictor
            del self.est.refiner # delete the old score and refine predictors
            del self.est.glctx # delete the old glctx
            del self.est # delete the old estimator
            self.get_logger().info(f"Deleted old estimator")
            # clear cuda memory ?
            torch.cuda.empty_cache()

            # Load mesh and compute bounds
            mesh = trimesh.load(self.mesh_file, force="mesh", skip_materials=True)
            self.to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
            self.get_logger().info(f"Mesh lodaed from {self.mesh_file} | Bounds: {self.bbox.flatten()}")
            
            # Load symmetry transforms (combined over x, y, z axes)
            symmetry_tfs = symmetry_tfs_from_angles(
                self.symmetry_x_angles, self.symmetry_y_angles, self.symmetry_z_angles
            )
            if symmetry_tfs is not None:
                self.get_logger().debug(f"Symmetry transforms: {symmetry_tfs.shape}")
            else:
                self.get_logger().debug(f"No symmetry transforms")

            # Initialize estimator
            self.get_logger().info("Initializing estimator...")
            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()
            glctx = dr.RasterizeCudaContext()
            self.est = FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=scorer,
                refiner=refiner,
                debug_dir=self.debug_dir,
                debug=self.debug,
                glctx=glctx,
                symmetry_tfs=symmetry_tfs,
                meshname=rn,
            )
            self.get_logger().info("FoundationPose estimator re-initialized")
            
            # Reset tracking so we detect the new object from scratch
            if self.current_phase in ("PoseTracking", "StartPoseTracking", "DetectingAgain"):
                self.current_phase = "DetectingAgain"
            self.initial_detection_counter = 0
            
            self.current_phase = "DetectingAgain"
            self._lock.release()
            self._reset_pose_filter("mesh/target object updated")
            return
        
        else:
            # just a change of target object        
            if self.seg_model_type == "yolo":
                if new_target not in list(DET_NAMES.values()):
                    self.get_logger().error(f"Ignoring target_object '{new_target}' (not a valid COCO class). Valid: {list(DET_NAMES.values())}")
                    return
                
            self.target_object = new_target
            self.get_logger().info(f"Target object changed to: {self.target_object}")
            
            # Reset tracking so we detect the new object from scratch
            if self.current_phase in ("PoseTracking", "StartPoseTracking", "DetectingAgain"):
                self.current_phase = "DetectingAgain"
            self.initial_detection_counter = 0
            self._reset_pose_filter("target object changed")

    def _publish_marker(self, pose_msg: PoseStamped):
        marker = Marker()
        marker.header = pose_msg.header
        marker.ns = "foundationpose"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose = pose_msg.pose
        scale = float(self.get_parameter("marker_mesh_scale").value)
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.mesh_resource = self._marker_mesh_resource
        marker.mesh_use_embedded_materials = bool(
            self.get_parameter("marker_mesh_use_embedded_materials").value
        )
        self._marker_pub.publish(marker)

    def _camera_info_cb(self, msg: CameraInfo):
        if self.K is not None:
            return
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if self.resize_factor != 1:
            self.K = self.K.copy()
            self.K[0, 0] /= self.resize_factor
            self.K[1, 1] /= self.resize_factor
            self.K[0, 2] /= self.resize_factor
            self.K[1, 2] /= self.resize_factor
            self.get_logger().debug(f"Resized camera intrinsics K by factor {self.resize_factor}: {self.K}")
        self.get_logger().info(f"Received camera intrinsics K: {self.K}")
        self.current_phase = "WaitingForRGBD"

    def _rgbd_cb(self, color_msg: CompressedImage, depth_msg: CompressedImage):
        self.get_logger().info("Received RGBD message")
        self.rgbd_frames_counter_received += 1
        # Skip if camera intrinsics K not received yet
        if self.K is None:
            self.get_logger().warn("Camera intrinsics K not received yet, skipping RGBD message")
            return
        
        if not self.is_on:
            self.get_logger().info("Node is off, skipping RGBD message")
            self._reset_pose_filter("node is off")
            return
        
        # Skip if already processing something
        if self._lock.acquire(blocking=False):
            if self._processing:
                self._lock.release()
                self.get_logger().warn(f"Already processing something (state is {self.current_phase})... skipping RGBD message")
                return
            self._processing = True
            self._lock.release()
        else:
            self.get_logger().warn("Could not acquire lock, skipping RGBD message")
            return
        
        # Update current phase
        try:
            if "compressed" in self.get_parameter("color_topic").value:
                color = decode_compressed_color(color_msg)
            else:
                color = color_msg.data
            if "compressed" in self.get_parameter("depth_topic").value:
                depth = decode_compressed_depth(depth_msg, self.depth_scale)
            else:
                depth = depth_msg.data
                
            self.get_logger().debug(f"Got images with size: RGB {color.shape}, Depth {depth.shape}")
                
            # Resize depth to match color image size
            if color.shape[:2] != depth.shape[:2]:
                depth = cv2.resize(
                    depth,
                    (color.shape[1], color.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                self.get_logger().debug(f"Resized depth to match color image size: RGB {color.shape}, Depth {depth.shape}")
                
            if self.resize_factor != 1:
                color = cv2.resize(
                    color,
                    (color.shape[1] // self.resize_factor, color.shape[0] // self.resize_factor),
                    interpolation=cv2.INTER_LINEAR,
                )
                depth = cv2.resize(
                    depth,
                    (depth.shape[1] // self.resize_factor, depth.shape[0] // self.resize_factor),
                    interpolation=cv2.INTER_NEAREST,
                )
                self.get_logger().debug(f"Resized RGBD image by factor {self.resize_factor} : RGB {color.shape}, Depth {depth.shape}")
                
        except ValueError as e:
            self.get_logger().error(str(e))
            self._lock.acquire()
            self._processing = False
            self._lock.release()
            return
        
        self.rgbd_frames_counter_processed += 1
        
        valid_pose = False
        if "Tracking" not in self.current_phase:
            self.current_phase = "Detecting"
            
            if self.seg_model_type == "sam3":
                color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                self.seg_model.set_image(color)
                results = self.seg_model(text=[self.target_object])
                if results is None:
                    self.get_logger().warn(f"No results from segmentation model for frame {self.rgbd_frames_counter_processed} (model type: {self.seg_model_type}, object: {self.target_object})")
                    self._note_pose_lost_for_filter()
                    self._lock.acquire()
                    self._processing = False
                    self._lock.release()
                    return
                target_masks = results[0].masks #.data.cpu().numpy()
                if target_masks is None:
                    self.get_logger().warn(f"No target masks from segmentation model for frame {self.rgbd_frames_counter_processed} (model type: {self.seg_model_type}, object: {self.target_object})")
                    self._note_pose_lost_for_filter()
                    self._lock.acquire()
                    self._processing = False
                    self._lock.release()
                    return
                
                found_obects = len(target_masks)
                if found_obects == 1:
                    self.initial_detection_counter = self.min_initial_detection_counter # directly set to min_initial_detection_counter to start tracking
                    target_mask = target_masks[0].data.cpu().numpy()
                    target_mask = target_mask[0,...].astype(np.uint8)
                    # print(type(target_mask))
                    # print(f"target_mask.shape: {target_mask.shape}, dtype: {target_mask.dtype}, min: {target_mask.min()}, max: {target_mask.max()}")
                    self.get_logger().info(f"Initial detection counter ({self.target_object}): {self.initial_detection_counter} / {self.min_initial_detection_counter}")
                elif found_obects > 1:
                    self.get_logger().warn(f"Multiple objects found ({found_obects}) in frame {self.rgbd_frames_counter_processed}, cannot chose")
                    self.initial_detection_counter = 0
                else:
                    self.initial_detection_counter = 0

            elif self.seg_model_type == "yolo":
                # perform detection or initial pose estimation
                target_mask = None
                found_object = 0
                results = self.seg_model.track(color) # per image (if batching)
                for iter, result in enumerate(results):
                    if len(result.boxes) == 0 or result.boxes.id is None:
                        self.get_logger().warn(f"No boxes found in frame {self.rgbd_frames_counter_processed}, iter {iter}")
                        continue
                    class_ids = result.boxes.cls.cpu().numpy()
                    class_names = [DET_NAMES.get(cls_id, f"class_{cls_id}") for cls_id in class_ids]
                    scores = result.boxes.conf.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy()
                    masks = result.masks.data.cpu().numpy()
                    print(f"\n ===== [{self.rgbd_frames_counter_processed}] {iter} =====")
                    for cls_name, score, track_id, mask in zip(class_names, scores, track_ids, masks):
                        print(f"\t{cls_name} ({score:.2f}) {int(track_id)}")
                        if cls_name == self.target_object:
                            target_mask = mask
                            found_object += 1
                
                if found_object == 1:
                    # need min_initial_detection_counter detections in a row to start tracking
                    self.initial_detection_counter += 1
                    self.get_logger().info(f"Initial detection counter ({self.target_object}): {self.initial_detection_counter} / {self.min_initial_detection_counter}")
                elif found_object > 1:
                    self.get_logger().warn(f"Multiple objects found ({found_object}) in frame {self.rgbd_frames_counter_processed}, iter {iter} cannot chose")
                    self.initial_detection_counter = 0
                else:
                    # set or reset to 0 if not found
                    self.initial_detection_counter = 0
                
            if self.initial_detection_counter >= self.min_initial_detection_counter:
                self.initial_detection_counter = 0
                self.current_phase = "PoseEstimation"
                est_timer_start = time.time()
                # print(f"target_mask.shape: {target_mask.shape}, dtype: {target_mask.dtype}, min: {target_mask.min()}, max: {target_mask.max()}")
                target_mask = cv2.resize(target_mask, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
                target_mask = (target_mask > 0).astype(bool)
                pose = self.est.register(
                    K=self.K,
                    rgb=color,
                    depth=depth,
                    ob_mask=target_mask,
                    iteration=self.est_refine_iter,
                )
                valid_pose = True # always True for now
                est_timer_end = time.time()
                if self.enable_pose_tracking:
                    # if not enabled, we will just go back to running again detections and pose estimation
                    self.current_phase = "StartPoseTracking"
                    
                self.get_logger().info(f"Pose estimation time: {est_timer_end - est_timer_start:.3f} seconds")
                self.get_logger().info(f"Starting tracking after {self.initial_detection_counter} initial detections with {self.target_object}")
                
            
        elif self.current_phase == "PoseTracking" or self.current_phase == "StartPoseTracking":
            self.current_phase = "PoseTracking"
            # perform tracking
            print("============ PoseTracking =============")
            track_timer_start = time.time()
            pose = self.est.track_one(
                rgb=color,
                depth=depth,
                K=self.K,
                iteration=self.track_refine_iter,
            )
            valid_pose = True # always True for now
            track_timer_end = time.time()
            self.get_logger().info(f"Tracking time: {track_timer_end - track_timer_start:.3f} seconds")

        if valid_pose:
            # center_pose = pose@np.linalg.inv(self.to_origin)
            center_pose = pose

            # Convert pose to object coordinates
            R_cam = center_pose[:3, :3]
            t_cam = center_pose[:3, 3]

            
            R_cam = apply_rotate_z_in(R_cam, self.apply_z_in)
            R_cam = apply_rotate_y_in(R_cam, self.apply_y_in)
            R_cam = apply_rotate_x_in(R_cam, self.apply_x_in)

            # Optional constraints directly on yaw/pitch/roll at R_cam level (zxy convention)
            r_cam = Rotation.from_matrix(R_cam)
            yaw_cam, pitch_cam, roll_cam = r_cam.as_euler('zxy', degrees=True)

            if self.constraint_yaw_in == 0 and self.constraint_pitch_in == 0 and self.constraint_roll_in == 0:
                # force yaw, pitch, roll to 0
                R_cam = np.eye(3)
                r_cam = Rotation.from_matrix(R_cam)
                yaw_cam, pitch_cam, roll_cam = r_cam.as_euler('zxy', degrees=True)
            else:
                # roll
                if self.constraint_roll_in is not None:
                    if isinstance(self.constraint_roll_in, (int, float)) and float(self.constraint_roll_in) == 0.0:
                        target_roll = 0.0
                    else:
                        lo = float(self.constraint_roll_in[0])
                        hi = float(self.constraint_roll_in[1])
                        width = hi - lo
                        if width <= 0:
                            raise ValueError(f"constraint_roll_in must be a valid range [lo, hi] with hi>lo, got {self.constraint_roll_in!r}")
                        roll_360 = float(np.mod(roll_cam, 360.0))
                        target_roll = float(np.mod(roll_360 - lo, width) + lo)
                    delta_roll = target_roll - roll_cam
                    R_cancel_roll = Rotation.from_euler('y', delta_roll, degrees=True).as_matrix()
                    R_cam = R_cam @ R_cancel_roll
                    r_cam = Rotation.from_matrix(R_cam)
                    yaw_cam, pitch_cam, roll_cam = r_cam.as_euler('zxy', degrees=True)

                # pitch
                if self.constraint_pitch_in is not None:
                    if isinstance(self.constraint_pitch_in, (int, float)) and float(self.constraint_pitch_in) == 0.0:
                        target_pitch = 0.0
                    else:
                        lo = float(self.constraint_pitch_in[0])
                        hi = float(self.constraint_pitch_in[1])
                        width = hi - lo
                        if width <= 0:
                            raise ValueError(f"constraint_pitch_in must be a valid range [lo, hi] with hi>lo, got {self.constraint_pitch_in!r}")
                        pitch_360 = float(np.mod(pitch_cam, 360.0))
                        target_pitch = float(np.mod(pitch_360 - lo, width) + lo)
                    delta_pitch = target_pitch - pitch_cam
                    R_cancel_pitch = Rotation.from_euler('x', delta_pitch, degrees=True).as_matrix()
                    R_cam = R_cam @ R_cancel_pitch
                    r_cam = Rotation.from_matrix(R_cam)
                    yaw_cam, pitch_cam, roll_cam = r_cam.as_euler('zxy', degrees=True)

                # yaw
                if self.constraint_yaw_in is not None:
                    if isinstance(self.constraint_yaw_in, (int, float)) and float(self.constraint_yaw_in) == 0.0:
                        target_yaw = 0.0
                    else:
                        lo = float(self.constraint_yaw_in[0])
                        hi = float(self.constraint_yaw_in[1])
                        width = hi - lo
                        if width <= 0:
                            raise ValueError(f"constraint_yaw_in must be a valid range [lo, hi] with hi>lo, got {self.constraint_yaw_in!r}")
                        yaw_360 = float(np.mod(yaw_cam, 360.0))
                        target_yaw = float(np.mod(yaw_360 - lo, width) + lo)
                    delta_yaw = target_yaw - yaw_cam
                    R_cancel_yaw = Rotation.from_euler('z', delta_yaw, degrees=True).as_matrix()
                    R_cam = R_cam @ R_cancel_yaw
                    r_cam = Rotation.from_matrix(R_cam)
                    yaw_cam, pitch_cam, roll_cam = r_cam.as_euler('zxy', degrees=True)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = color_msg.header.stamp
            pose_msg.header.frame_id = self.pose_frame_id

            euler_cam = np.array([yaw_cam, pitch_cam, roll_cam], dtype=np.float64)
            self.get_logger().info(f"Pose: t = {t_cam}")
            self.get_logger().info(f"R = {R_cam}")
            self.get_logger().info(f"euler = {euler_cam}")
            self.get_logger().info(f"yaw = {yaw_cam:.2f} deg, pitch = {pitch_cam:.2f} deg, roll = {roll_cam:.2f} deg")

            new_r_cam = r_cam                 
            new_q_cam = new_r_cam.as_quat()

            # add optional filter
            if self.use_kalman_filter:
                stamp_s = float(color_msg.header.stamp.sec) + (float(color_msg.header.stamp.nanosec) * 1e-9)
                if self._pose_filter_last_stamp_s is None:
                    dt = 1e9  # forces initialization
                else:
                    dt = max(0.0, stamp_s - float(self._pose_filter_last_stamp_s))
                self._pose_filter_last_stamp_s = stamp_s

                self._pose_filter_lost_frames = 0
                self._pose_filter.update(
                    dt=dt,
                    meas_pos=t_cam,
                    meas_quat_xyzw=new_q_cam,
                    process_noise=float(self.get_parameter("pose_filter_process_noise").value),
                    meas_noise=float(self.get_parameter("pose_filter_meas_noise").value),
                    slerp_factor=float(self.get_parameter("pose_filter_slerp_factor").value),
                )
                t_cam = self._pose_filter.pos.copy()
                new_q_cam = self._pose_filter.quat_xyzw.copy()


            pose_msg.pose.position.x = float(t_cam[0])
            pose_msg.pose.position.y = float(t_cam[1])
            pose_msg.pose.position.z = float(t_cam[2])
            
            pose_msg.pose.orientation.x = float(new_q_cam[0])
            pose_msg.pose.orientation.y = float(new_q_cam[1])
            pose_msg.pose.orientation.z = float(new_q_cam[2])
            pose_msg.pose.orientation.w = float(new_q_cam[3])
            self._pose_pub.publish(pose_msg)
            self._publish_marker(pose_msg)
        else:
            self._note_pose_lost_for_filter()

        # Finish processing by releasing the lock
        self._lock.acquire()
        self._processing = False
        self._lock.release()
        

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except:
            pass
        super().destroy_node()

def main(args):
    rclpy.init()
    node = FoundationPoseROS2Node(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_file", type=str, default="./assets/juice/juice.obj", help="Path to object mesh file (e.g. juice.obj). Empty = use demo_data/bottle.")
    parser.add_argument("--target_object", type=str, default="bottle", help="Target object class name for YOLO (e.g. bottle, cup).")
    parser.add_argument("--est_refine_iter", type=int, default=5, help="Number of refinement iterations for registration.")
    parser.add_argument("--track_refine_iter", type=int, default=2, help="Number of refinement iterations for tracking.")
    parser.add_argument("--debug", type=int, default=1, help="Debug level.")
    parser.add_argument("--debug_dir", type=str, default="", help="Debug directory.")
    parser.add_argument("--depth_scale", type=float, default=0.001, help="Depth scale.")
    parser.add_argument("--camera_name", type=str, default="realsense_head_front", help="Camera name.")
    
    # paremters depending on the camera name
    # parser.add_argument("--color_topic", type=str, default="/rgbd/realsense_test/color/image_raw/compressed", help="Color topic.")
    # parser.add_argument("--depth_topic", type=str, default="/rgbd/realsense_test/aligned_depth_to_color/image_raw/compressedDepth", help="Depth topic.")
    # parser.add_argument("--camera_info_topic", type=str, default="/rgbd/realsense_test/color/camera_info", help="Camera info topic.")
    # parser.add_argument("--pose_frame_id", type=str, default="realsense_test_depth_optical_frame", help="Pose frame id.")
    
    parser.add_argument("--use_kalman_filter", "-kf", action="store_true", default=False, help="Use Kalman filter for pose estimation.")
    parser.add_argument("--slop", type=float, default=1.0, help="Slop.")
    parser.add_argument("--seg_model_type", type=str, default="yolo", help="Segmentation model type.")
    parser.add_argument("--seg_model_name", type=str, default="yolo26n-seg.pt", help="Segmentation model name.")
    parser.add_argument("--resize_factor", type=int, default=1, help="Resize factor to divide the image size by this factor.")
    parser.add_argument("--min_initial_detection_counter", type=int, default=5, help="Minimum initial detection counter.")
    parser.add_argument("--enable_pose_tracking", action="store_true", default=False, help="Enable pose tracking.")
    parser.add_argument("--symmetry_x_angles", "-sxa", type=str, default=None, help="Symmetry roll angles (about x). Format: 'a1,a2,a3,...'. Empty/None = no x symmetry transforms.")
    parser.add_argument("--symmetry_y_angles", "-sya", type=str, default=None, help="Symmetry pitch angles (about y). Format: 'a1,a2,a3,...'. Empty/None = no y symmetry transforms.")
    parser.add_argument("--symmetry_z_angles", "-sza", type=str, default=None, help="Symmetry yaw angles (about z). Format: 'yaw1,yaw2,yaw3,...'. Empty = no z symmetry transforms.")
    parser.add_argument("--fp_verbosity", "-v", type=str, default="info", help="Verbosity level for FoundationPose. Valid: debug, info, warning, error, critical.")
    args = parser.parse_args()
    
    args.color_topic = f"/rgbd/{args.camera_name}/color/image_raw/compressed"
    args.depth_topic = f"/rgbd/{args.camera_name}/aligned_depth_to_color/image_raw/compressedDepth"
    args.camera_info_topic = f"/rgbd/{args.camera_name}/aligned_depth_to_color/camera_info"
    args.pose_frame_id = f"{args.camera_name}_depth_optical_frame"
    
    main(args)

        
