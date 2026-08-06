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
  - /orchestrator/pose/toggle_tracking (std_msgs/Bool) to enable/disable pose tracking
  - /orchestrator/pose/target_object (std_msgs/String) to set the target object class at runtime

Publishes:
  - /foundation_pose/object_pose (geometry_msgs/PoseStamped)
  - /foundation_pose/object_marker (visualization_msgs/Marker) mesh marker at pose
  - /foundation_pose/mask_image (sensor_msgs/Image) optional debug mask overlay
"""

import argparse
import os
import time
import threading
from typing import Optional, Sequence

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker
from message_filters import Subscriber, ApproximateTimeSynchronizer

from foundationpose.estimater import *
from sensor_msgs.msg import Image as RosImage  # after estimater * (PIL.Image otherwise shadows)
from ultralytics import YOLO
from ultralytics.models.sam import SAM3SemanticPredictor

from scipy.spatial.transform import Rotation

# For YOLO based detection
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


OBJECT_KEYS_TO_PARAMETERS = {
    # flips on x and y at 180 !
    "baguette" : {"mesh_file": "./assets/hackathon3/baguette/baguette.obj",
                  "symmetry_x_angles": "0,180",
                  "symmetry_y_angles": "0,180",
                  "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
                  "target_object": "bread",
                  "constraint_yaw_in": 0,
                  "constraint_pitch_in": [0,180],
                  "constraint_roll_in": [0,180]},

    # seems okay and does not flip
    "banana" : {"mesh_file": "./assets/hackathon3/banana/banana.obj",
                "symmetry_x_angles": "",
                "symmetry_y_angles": "",
                "symmetry_z_angles": "",
                "target_object": "banana",
                "constraint_yaw_in": None,
                "constraint_pitch_in": None,
                "constraint_roll_in": None},

    # still flips up and down, adjusted size to retry
    "coffeecan" : {"mesh_file": "./assets/hackathon3/coffeecan/coffeecan.obj",
                   "symmetry_x_angles": "",
                   "symmetry_y_angles": "",
                   "symmetry_z_angles": "",
                   "target_object": "blue container",
                   "constraint_yaw_in": 0,
                   "constraint_pitch_in": None,
                   "constraint_roll_in": None},

    # lots of flipping difficult to stabilize
    "egg" : {"mesh_file": "./assets/hackathon3/egg/egg.obj",
             "symmetry_x_angles": "0,180",
             "symmetry_y_angles": "0,180",
             "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
             "target_object": "egg",
             "constraint_yaw_in": 0,
             "constraint_pitch_in": 0,
             "constraint_roll_in": 0},

    # good it seems that does not flip up/down but the handle does not always get oriented correctly
    "flowercup" : {"mesh_file": "./assets/hackathon3/flowercup/flowercup.obj",
                   "symmetry_x_angles": "",
                   "symmetry_y_angles": "",
                   "symmetry_z_angles": "",
                   "target_object": "yellow mug",
                   "constraint_yaw_in": None,
                   "constraint_pitch_in": None,
                   "constraint_roll_in": None},

    # not detected with keyword jam
    "jam" : {"mesh_file": "./assets/hackathon3/jam/jam.obj",
             "symmetry_x_angles": "",
             "symmetry_y_angles": "",
             "symmetry_z_angles": "",
             "target_object": "orange jam",
             "constraint_yaw_in": None,
             "constraint_pitch_in": None,
             "constraint_roll_in": None},

    # ok
    "milk" : {"mesh_file": "./assets/hackathon3/milk/milk.obj",
              "symmetry_x_angles": "",
              "symmetry_y_angles": "",
              "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
              "target_object": "white bottle",
              "constraint_yaw_in": 0,
              "constraint_pitch_in": None,
              "constraint_roll_in": None},

    # hard to find but seems good when found... investigate, maybe a bigger one !
    "minicheese" : {"mesh_file": "./assets/hackathon3/minicheese/minicheese.obj",
                    "symmetry_x_angles": "",
                    "symmetry_y_angles": "",
                    "symmetry_z_angles": "",
                    "target_object": "triangle cheese",
                    "constraint_yaw_in": None,
                    "constraint_pitch_in": None,
                    "constraint_roll_in": None},

    # seems robust could be a good anchor point
    "pan" : {"mesh_file": "./assets/hackathon3/pan/pan.obj",
             "symmetry_x_angles": "",
             "symmetry_y_angles": "",
             "symmetry_z_angles": "",
             "target_object": "pan",
             "constraint_yaw_in": None,
             "constraint_pitch_in": None,
             "constraint_roll_in": None},

    # problem gets a lot of multiple objects + rotations
    "redapple" : {"mesh_file": "./assets/hackathon3/redapple/redapple.obj",
                  "symmetry_x_angles": "0,180",
                  "symmetry_y_angles": "0,180",
                  "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
                  "target_object": "red apple",
                  "constraint_yaw_in": 0,
                  "constraint_pitch_in": 0,
                  "constraint_roll_in": 0},

    "smallmilk" : {"mesh_file": "./assets/hackathon3/smallmilk/smallmilk.obj",
                   "symmetry_x_angles": "",
                   "symmetry_y_angles": "",
                   "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
                   "target_object": "white bottle",
                   "constraint_yaw_in": 0,
                   "constraint_pitch_in": None,
                   "constraint_roll_in": None},

    "smallsanpellegrino" : {"mesh_file": "./assets/hackathon3/smallsanpellegrino/smallsanpellegrino.obj",
                            "symmetry_x_angles": "",
                            "symmetry_y_angles": "",
                            "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
                            "target_object": "green bottle",
                            "constraint_yaw_in": 0,
                            "constraint_pitch_in": None,
                            "constraint_roll_in": None},

    # hard to detect and flips on several axes
    "spam" : {"mesh_file": "./assets/hackathon3/spam/spam.obj",
              "symmetry_x_angles": "",
              "symmetry_y_angles": "",
              "symmetry_z_angles": "",
              "target_object": "blue container",
              "constraint_yaw_in": [0,180],
              "constraint_pitch_in": [0,180],
              "constraint_roll_in": [0,180]},

    # okay but flips on one axis
    "ycbmustard" : {"mesh_file": "./assets/hackathon3/ycbmustard/ycbmustard.obj",
                    "symmetry_x_angles": "",
                    "symmetry_y_angles": "",
                    "symmetry_z_angles": "",
                    "target_object": "yellow bottle",
                    "constraint_yaw_in": [0,180],
                    "constraint_pitch_in": None,
                    "constraint_roll_in": None},

    "juice" : { "mesh_file": "./assets/hackathon2/juice/juice.obj",
                "symmetry_x_angles": "",
                "symmetry_y_angles": "",
                "symmetry_z_angles": "",
                "target_object": "carton bottle",
                "constraint_yaw_in": [0,180],
                "constraint_pitch_in": None,
                "constraint_roll_in": None},

    }
class FoundationPoseROS2Node(Node):
    def __init__(self, args):
        super().__init__("foundation_pose_node")

        # RCUTILS_LOGGING_SEVERITY alone is unreliable for node loggers when using
        # python node.py (rclpy.init gets no --ros-args). Set level explicitly.
        ros_verbosity = (args.ros_verbosity or os.environ.get("RCUTILS_LOGGING_SEVERITY") or "INFO").upper()
        severity_map = {
            "DEBUG": LoggingSeverity.DEBUG,
            "INFO": LoggingSeverity.INFO,
            "WARN": LoggingSeverity.WARN,
            "WARNING": LoggingSeverity.WARN,
            "ERROR": LoggingSeverity.ERROR,
            "FATAL": LoggingSeverity.FATAL,
        }
        if ros_verbosity not in severity_map:
            raise ValueError(f"Invalid ros_verbosity: {ros_verbosity}. Valid: {list(severity_map)}")
        self.get_logger().set_level(severity_map[ros_verbosity])

        # Declare ROS parameters
        self.declare_parameter("object_key", args.object_key)
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
        # Set current code directory
        code_dir = os.path.dirname(os.path.realpath(__file__))

        # Get parameters
        self.object_key = self.get_parameter("object_key").value
        self.mesh_file = self.get_parameter("mesh_file").value
        assert(os.path.exists(self.mesh_file)), f"Mesh file {self.mesh_file} does not exist"
        mesh_file_basename = os.path.basename(self.mesh_file)
        mesh_file_rn = mesh_file_basename.split(".")[0]
        self._marker_mesh_resource = f"file:///mesh_assets/{mesh_file_rn}/{mesh_file_basename}"

        _abs_mesh = os.path.normpath(os.path.abspath(self.mesh_file))
        self.constraint_yaw_in = None
        self.constraint_pitch_in = None
        self.constraint_roll_in = None
        for params in OBJECT_KEYS_TO_PARAMETERS.values():
            if mesh_file_basename in params["mesh_file"]:
                self.constraint_yaw_in = params.get("constraint_yaw_in")
                self.constraint_pitch_in = params.get("constraint_pitch_in")
                self.constraint_roll_in = params.get("constraint_roll_in")
                break


        # Get debug directory and create if it doesn't exist
        self.debug_dir = self.get_parameter("debug_dir").value
        if not self.debug_dir:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.debug_dir = f"{code_dir}/debug_node/{timestamp}_{args.object_key}"

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
        self.publish_mask_image = args.publish_mask_image
        self.seg_model_type = args.seg_model_type
        self.symmetry_x_angles = args.symmetry_x_angles
        self.symmetry_y_angles = args.symmetry_y_angles
        self.symmetry_z_angles = args.symmetry_z_angles
        self.fp_verbosity = args.fp_verbosity
        self.use_onnx = args.use_onnx
        self.refiner_onnx = args.refiner_onnx
        self.scorer_onnx = args.scorer_onnx
        self.prefer_tensorrt = not args.no_prefer_tensorrt
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
        self.get_logger().debug(f"Object key: {self.object_key}")
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
        self.get_logger().debug(f"Publish mask image: {self.publish_mask_image}")
        self.get_logger().debug(f"Symmetry x angles: {self.symmetry_x_angles}")
        self.get_logger().debug(f"Symmetry y angles: {self.symmetry_y_angles}")
        self.get_logger().debug(f"Symmetry z angles: {self.symmetry_z_angles}")
        self.get_logger().debug(f"Constraint yaw in: {self.constraint_yaw_in}")
        self.get_logger().debug(f"Constraint pitch in: {self.constraint_pitch_in}")
        self.get_logger().debug(f"Constraint roll in: {self.constraint_roll_in}")
        self.get_logger().debug(f"Use onnx: {self.use_onnx}")
        self.get_logger().debug(f"Prefer tensorrt: {self.prefer_tensorrt}")
        self.get_logger().debug(f"Refiner onnx: {self.refiner_onnx}")
        self.get_logger().debug(f"Scorer onnx: {self.scorer_onnx}")

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

        # Set logger and seed (for estimater, not for ROS get_logger())
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
                imgsz=644,
                task="segment",
                mode="predict",
                model=f"sam3/{self.seg_model_name}",
                half=True,  # Use FP16 for faster inference
                save=False,
                verbose=False,
            )
            self.seg_model = SAM3SemanticPredictor(overrides=overrides)
            # run a fake pass to warm up the model
            self.seg_model.set_image(np.zeros((1080, 1920, 3), dtype=np.uint8))
            self.seg_model(text=[self.target_object], verbose=False)
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
        if self.use_onnx:
            self.get_logger().info(f"Using ONNX predictors (prefer_tensorrt: {self.prefer_tensorrt})")
            from foundationpose.onnx_predictors import PoseRefinePredictorOnnx, ScorePredictorOnnx
            scorer_kwargs = {"prefer_tensorrt": self.prefer_tensorrt}
            refiner_kwargs = {"prefer_tensorrt": self.prefer_tensorrt}
            if self.scorer_onnx:
                scorer_kwargs["onnx_path"] = self.scorer_onnx
            if self.refiner_onnx:
                refiner_kwargs["onnx_path"] = self.refiner_onnx
            scorer = ScorePredictorOnnx(**scorer_kwargs)
            refiner = PoseRefinePredictorOnnx(**refiner_kwargs)
        else:
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

        # QoS PROFILES
        qos_pose = qos_profile_sensor_data
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=15,
        )

        # Commands: RELIABLE, Queue 10. Guarantees mesh loading commands aren't dropped.
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # State: RELIABLE, Queue 1. Guarantees delivery, but only keeps the freshest state.
        qos_state = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_marker = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # PUBLISHERS & SUBSCRIBERS
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._camera_info_cb,
            qos_state, # Camera info can use state profile
        )

        self._pose_pub = self.create_publisher(
            PoseStamped,
            "/foundation_pose/object_pose",
            qos_pose,
        )

        self._marker_pub = self.create_publisher(
            Marker,
            "/foundation_pose/object_marker",
            qos_marker,
        )

        self._mesh_status_pub = self.create_publisher(
            Bool,
            "/foundation_pose/mesh_status",
            qos_state,
        )

        self._mask_image_pub = None
        if self.publish_mask_image:
            self._mask_image_pub = self.create_publisher(
                RosImage,
                "/foundation_pose/mask_image",
                qos_sensor,
            )

        self._toggle_fp_sub = self.create_subscription(
            Bool,
            "/orchestrator/foundation_pose/toggle",
            self._toggle_fp_cb,
            qos_state,
        )

        self._toggle_tracking_sub = self.create_subscription(
            Bool,
            "/orchestrator/pose/toggle_tracking",
            self._toggle_tracking_cb,
            qos_state,
        )

        self._target_object_sub = self.create_subscription(
            String,
            "/orchestrator/foundation_pose/target_object",
            self._target_object_cb,
            qos_cmd,
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

    def _toggle_fp_cb(self, msg: Bool):
        prev = self.is_on
        self.is_on = msg.data
        self._prev_is_on = prev
        self.get_logger().info(f"FoundationPose toggled: is_on = {self.is_on}")

        if msg.data == False:
            if self.current_phase == "PoseTracking" or self.current_phase == "StartPoseTracking":
                self.get_logger().info("Stopping pose tracking back to detecting for later")
                self.current_phase = "DetectingAgain"

    def _toggle_tracking_cb(self, msg: Bool):
        self.enable_pose_tracking = msg.data
        self.get_logger().info(f"Pose tracking toggled: enable_pose_tracking = {self.enable_pose_tracking}")

        if msg.data == False:
            if self.current_phase == "PoseTracking" or self.current_phase == "StartPoseTracking":
                self.get_logger().info("Stopping pose tracking back to detecting")
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
            if self.use_onnx:
                from foundationpose.onnx_predictors import PoseRefinePredictorOnnx, ScorePredictorOnnx
                scorer_kwargs = {"prefer_tensorrt": self.prefer_tensorrt}
                refiner_kwargs = {"prefer_tensorrt": self.prefer_tensorrt}
                if self.scorer_onnx:
                    scorer_kwargs["onnx_path"] = self.scorer_onnx
                if self.refiner_onnx:
                    refiner_kwargs["onnx_path"] = self.refiner_onnx
                scorer = ScorePredictorOnnx(**scorer_kwargs)
                refiner = PoseRefinePredictorOnnx(**refiner_kwargs)
            else:
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

            self._mesh_status_pub.publish(Bool(data=True))

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

    def _publish_mask_image(self, rgb, masks, header):
        """Overlay one color per mask on rgb (HxWx3) and publish as raw Image."""
        if self._mask_image_pub is None or masks is None or len(masks) == 0:
            return
        t0 = time.time()
        vis = rgb.copy()
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        ]
        for i, mask in enumerate(masks):
            m = np.asarray(mask)
            if m.ndim == 3:
                m = m[0]
            if m.shape[:2] != vis.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
            sel = m.astype(bool)
            if not sel.any():
                continue
            color = np.array(colors[i % len(colors)], dtype=np.float32)
            vis[sel] = (0.45 * vis[sel].astype(np.float32) + 0.55 * color).astype(np.uint8)

        vis = np.ascontiguousarray(vis)
        msg = RosImage()
        msg.header = header
        msg.height = vis.shape[0]
        msg.width = vis.shape[1]
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = vis.shape[1] * 3
        msg.data = vis.tobytes()
        self._mask_image_pub.publish(msg)
        self.get_logger().debug(f"Publish mask image: {(time.time()-t0)*1000:.2f} ms ({len(masks)} mask(s))")

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
        self.get_logger().debug("Received RGBD message")
        self.rgbd_frames_counter_received += 1
        # Skip if camera intrinsics K not received yet
        if self.K is None:
            self.get_logger().warn("Camera intrinsics K not received yet, skipping RGBD message")
            return

        if not self.is_on:
            self.get_logger().info("Node is off, skipping RGBD message")
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
                # color is already RGB from decode_compressed_color
                self.seg_model.set_image(color)
                results = self.seg_model(text=[self.target_object], verbose=False)
                
                # Log SAM3 inference speed and results
                n_masks = len(results[0].masks) if results is not None and results[0].masks is not None else 0
                speed = results[0].speed if results is not None else {}
                preprocess_latency = speed.get("preprocess", 0)
                inference_latency = speed.get("inference", 0)
                postprocess_latency = speed.get("postprocess", 0)
                total_latency = preprocess_latency + inference_latency + postprocess_latency
                self.get_logger().info(
                    f"Frame {self.rgbd_frames_counter_processed}: SAM3 found {n_masks} object(s), latency: {total_latency:.2f} ms"
                )
                self.get_logger().debug(
                    f"Frame {self.rgbd_frames_counter_processed}: SAM3 speed: "
                    f"preprocess={preprocess_latency:.2f}ms, inference={inference_latency:.2f}ms, postprocess={postprocess_latency:.2f}ms"
                )
           
                if results is None:
                    self.get_logger().warn(f"No results from segmentation model for frame {self.rgbd_frames_counter_processed} (model type: {self.seg_model_type}, object: {self.target_object})")
                    self._lock.acquire()
                    self._processing = False
                    self._lock.release()
                    return
                target_masks = results[0].masks #.data.cpu().numpy()
                if target_masks is None:
                    self.get_logger().error(f"No target masks from segmentation model for frame {self.rgbd_frames_counter_processed} (model type: {self.seg_model_type}, object: {self.target_object})")
                    self._lock.acquire()
                    self._processing = False
                    self._lock.release()
                    return

                found_obects = len(target_masks)
                if self.publish_mask_image and found_obects > 0:
                    masks_list = [target_masks[i].data.cpu().numpy()[0] for i in range(found_obects)]
                    self._publish_mask_image(color, masks_list, color_msg.header)
                if found_obects == 1:
                    self.initial_detection_counter = self.min_initial_detection_counter # directly set to min_initial_detection_counter to start tracking
                    target_mask = target_masks[0].data.cpu().numpy()
                    target_mask = target_mask[0,...].astype(np.uint8)
                    # print(type(target_mask))
                    # print(f"target_mask.shape: {target_mask.shape}, dtype: {target_mask.dtype}, min: {target_mask.min()}, max: {target_mask.max()}")
                    self.get_logger().debug(f"Initial detection counter ({self.target_object}): {self.initial_detection_counter} / {self.min_initial_detection_counter}")
                elif found_obects > 1:
                    self.get_logger().warn(f"Multiple objects found ({found_obects}) in frame {self.rgbd_frames_counter_processed}, cannot chose")
                    self.initial_detection_counter = 0
                else:
                    self.initial_detection_counter = 0

            elif self.seg_model_type == "yolo":
                # perform detection or initial pose estimation
                target_mask = None
                found_object = 0
                target_masks_list = []
                results = self.seg_model.track(
                    color,
                    verbose=False,
                ) # per image (if batching)
                n_boxes = len(results) if results is not None else 0
                self.get_logger().info(f"Frame {self.rgbd_frames_counter_processed}: YOLO found {n_boxes} object(s)")
                for iter, result in enumerate(results):
                    if len(result.boxes) == 0 or result.boxes.id is None:
                        self.get_logger().warn(f"No boxes found in frame {self.rgbd_frames_counter_processed}, iter {iter}")
                        continue
                    class_ids = result.boxes.cls.cpu().numpy()
                    class_names = [DET_NAMES.get(cls_id, f"class_{cls_id}") for cls_id in class_ids]
                    scores = result.boxes.conf.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy()
                    masks = result.masks.data.cpu().numpy()
                    self.get_logger().debug(f"\n Found boxes in frame {self.rgbd_frames_counter_processed}, iter {iter}")
                    for cls_name, score, track_id, mask in zip(class_names, scores, track_ids, masks):
                        self.get_logger().debug(f"\t{cls_name} ({score:.2f}) {int(track_id)}")
                        if cls_name == self.target_object:
                            target_mask = mask
                            found_object += 1
                            target_masks_list.append(mask)

                if self.publish_mask_image and target_masks_list:
                    self._publish_mask_image(color, target_masks_list, color_msg.header)

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
                    ros_logger=self.get_logger(),
                )
                valid_pose = True # always True for now
                est_timer_end = time.time()
                self.get_logger().info(f"Frame {self.rgbd_frames_counter_processed}: Pose estimation time {(est_timer_end - est_timer_start)*1000:.2f} ms")
                
                if self.enable_pose_tracking:
                    # if not enabled, we will just go back to running again detections and pose estimation
                    self.current_phase = "StartPoseTracking"
                    self.get_logger().info(f"Starting tracking with {self.target_object}")


        elif self.current_phase == "PoseTracking" or self.current_phase == "StartPoseTracking":
            self.current_phase = "PoseTracking"
            # perform tracking
            self.get_logger().info("Starting pose tracking")
            track_timer_start = time.time()
            pose = self.est.track_one(
                rgb=color,
                depth=depth,
                K=self.K,
                iteration=self.track_refine_iter,
                ros_logger=self.get_logger(),
            )
            valid_pose = True # always True for now
            track_timer_end = time.time()
            self.get_logger().info(f"Frame {self.rgbd_frames_counter_processed}: Tracking time {(track_timer_end - track_timer_start)*1000:.2f} ms")

        if valid_pose:
            # center_pose = pose@np.linalg.inv(self.to_origin)
            center_pose = pose

            # Convert pose to object coordinates
            R_cam = center_pose[:3, :3]
            t_cam = center_pose[:3, 3]

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
            self.get_logger().debug(f"Pose: t = {t_cam}")
            self.get_logger().debug(f"R = {R_cam}")
            self.get_logger().debug(f"euler = {euler_cam}")
            self.get_logger().debug(f"yaw = {yaw_cam:.2f} deg, pitch = {pitch_cam:.2f} deg, roll = {roll_cam:.2f} deg")

            new_r_cam = r_cam
            new_q_cam = new_r_cam.as_quat()

            pose_msg.pose.position.x = float(t_cam[0])
            pose_msg.pose.position.y = float(t_cam[1])
            pose_msg.pose.position.z = float(t_cam[2])

            pose_msg.pose.orientation.x = float(new_q_cam[0])
            pose_msg.pose.orientation.y = float(new_q_cam[1])
            pose_msg.pose.orientation.z = float(new_q_cam[2])
            pose_msg.pose.orientation.w = float(new_q_cam[3])
            self._pose_pub.publish(pose_msg)
            self._publish_marker(pose_msg)
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
    parser.add_argument("--object_key", type=str, choices=sorted(OBJECT_KEYS_TO_PARAMETERS.keys()), default="milk", help="Object keyword from OBJECT_KEYS_TO_PARAMETERS.")
    parser.add_argument("--est_refine_iter", type=int, default=5, help="Number of refinement iterations for registration.")
    parser.add_argument("--track_refine_iter", type=int, default=2, help="Number of refinement iterations for tracking.")
    parser.add_argument("--debug", type=int, default=1, help="Debug level.")
    parser.add_argument("--debug_dir", type=str, default="", help="Debug directory.")
    parser.add_argument("--depth_scale", type=float, default=0.001, help="Depth scale.")
    parser.add_argument("--camera_name", type=str, default="realsense_head_front", help="Camera name.")
    parser.add_argument("--slop", type=float, default=1.0, help="Slop.")
    parser.add_argument("--seg_model_type", type=str, default="yolo", help="Segmentation model type.")
    parser.add_argument("--seg_model_name", type=str, default="yolo26n-seg.pt", help="Segmentation model name.")
    parser.add_argument("--resize_factor", type=int, default=1, help="Resize factor to divide the image size by this factor.")
    parser.add_argument("--min_initial_detection_counter", type=int, default=5, help="Minimum initial detection counter.")
    parser.add_argument("--enable_pose_tracking", action="store_true", default=False, help="Enable pose tracking.")
    parser.add_argument("--publish_mask_image", action="store_true", default=False, help="Publish raw RGB Image with colored detection masks for RViz debug.")
    parser.add_argument("--fp_verbosity", type=str, default="warning", help="Verbosity level for FoundationPose. Valid: debug, info, warning, error, critical.")
    parser.add_argument("--ros_verbosity", type=str, default="info", help="ROS logger severity (debug/info/warn/error/fatal). Defaults to RCUTILS_LOGGING_SEVERITY or INFO.")
    parser.add_argument("--use_onnx", action="store_true", default=False, help="Use ONNX predictors instead of the default scorer and refiner.")
    parser.add_argument("--refiner_onnx", type=str, default="", help="Optional path to refiner_net.onnx.")
    parser.add_argument("--scorer_onnx", type=str, default="", help="Optional path to score_net.onnx.")
    parser.add_argument("--no_prefer_tensorrt", action="store_true", default=False, help="Disable TensorRT EP when using ONNX predictors.")
    args = parser.parse_args()

    # Set parameters based on the object key
    args.mesh_file = OBJECT_KEYS_TO_PARAMETERS[args.object_key]["mesh_file"]
    args.target_object = OBJECT_KEYS_TO_PARAMETERS[args.object_key]["target_object"]
    args.symmetry_x_angles = OBJECT_KEYS_TO_PARAMETERS[args.object_key]["symmetry_x_angles"]
    args.symmetry_y_angles = OBJECT_KEYS_TO_PARAMETERS[args.object_key]["symmetry_y_angles"]
    args.symmetry_z_angles = OBJECT_KEYS_TO_PARAMETERS[args.object_key]["symmetry_z_angles"]

    # Set the parameters based on the camera name
    args.color_topic = f"/rgbd/{args.camera_name}/color/image_raw/compressed"
    args.depth_topic = f"/rgbd/{args.camera_name}/aligned_depth_to_color/image_raw/compressedDepth"
    args.camera_info_topic = f"/rgbd/{args.camera_name}/aligned_depth_to_color/camera_info"
    args.pose_frame_id = f"{args.camera_name}_depth_optical_frame"

    main(args)

