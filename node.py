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

Publishes:
  - object_pose (geometry_msgs/PoseStamped)
"""

import os
import time
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer

from foundationpose.estimater import *
from ultralytics import YOLO

from scipy.spatial.transform import Rotation

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

# Camera -> World axis conversion
R_wc = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0]
])

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

        # Set current code directory
        code_dir = os.path.dirname(os.path.realpath(__file__))

        # Get parameters
        self.mesh_file = self.get_parameter("mesh_file").value
        assert(os.path.exists(self.mesh_file)), f"Mesh file {self.mesh_file} does not exist"

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
        
        self.K = None # to be set by camera info callback
        self.est = None # to be set by estimator initialization
        self.current_phase = "NotInitialized"
        self.pose_last = None
        self.to_origin = None
        self.bbox = None
        self.frame_count = 0
        self._lock = threading.Lock()
        self._processing = False
        
        self.initial_detection_counter = 0
        
        self.rgbd_frames_counter_received = 0
        self.rgbd_frames_counter_processed = 0
        
        # Set logger and seed (for estimater)
        set_logging_format()
        set_seed(0)

        # Load mesh and compute bounds
        mesh = trimesh.load(self.mesh_file)
        self.to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        self.get_logger().info(f"Mesh lodaed from {self.mesh_file} | Bounds: {self.bbox.flatten()}")
        
        # Initialize segmentation / detection model
        self.get_logger().info(f"Initializing segmentation model {self.seg_model_name}...")
        self.seg_model = YOLO(self.seg_model_name)
        self.get_logger().info(f"Segmentation model {self.seg_model_name} initialized")
        
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
        
        # Skip if already processing something
        if self._lock.acquire(blocking=False):
            if self._processing:
                self._lock.release()
                self.get_logger().warn(f"Already processing something (state is {self.current_phase})... skipping RGBD message")
                return
            self._processing = True
            self._lock.release()
        else:
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
        if self.current_phase != "PoseTracking":
            self.current_phase = "Detecting"
            
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
                    self.current_phase = "PoseTracking"
                    
                self.get_logger().info(f"Pose estimation time: {est_timer_end - est_timer_start:.3f} seconds")
                self.get_logger().info(f"Starting tracking after {self.initial_detection_counter} initial detections with {self.target_object}")
                
                
        elif self.current_phase == "PoseTracking":
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
            # Convert pose to object coordinates
            R = R_wc @ pose[:3, :3]
            t = pose[:3, 3]
            r = Rotation.from_matrix(R)
            roll_deg, pitch_deg, yaw_deg = r.as_euler('xyz', degrees=True)
            self.get_logger().info(f"Pose: t = {t}, yaw = {yaw_deg:.2f} deg, pitch = {pitch_deg:.2f} deg, roll = {roll_deg:.2f} deg")

            pose_msg = PoseStamped()
            pose_msg.header.stamp = color_msg.header.stamp
            pose_msg.header.frame_id = self.pose_frame_id
            pose_msg.pose.position.x = float(t[0])
            pose_msg.pose.position.y = float(t[1])
            pose_msg.pose.position.z = float(t[2])
            q = r.as_quat()
            pose_msg.pose.orientation.x = float(q[0])
            pose_msg.pose.orientation.y = float(q[1])
            pose_msg.pose.orientation.z = float(q[2])
            pose_msg.pose.orientation.w = float(q[3])
            self._pose_pub.publish(pose_msg)
            
        
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
    parser.add_argument("--mesh_file", type=str, default="./assets/bottle/ref_mesh.obj", help="Path to object mesh file (e.g. ref_mesh.obj). Empty = use demo_data/bottle.")
    parser.add_argument("--target_object", type=str, default="bottle", help="Target object class name for YOLO (e.g. bottle, cup).")
    parser.add_argument("--est_refine_iter", type=int, default=5, help="Number of refinement iterations for registration.")
    parser.add_argument("--track_refine_iter", type=int, default=2, help="Number of refinement iterations for tracking.")
    parser.add_argument("--debug", type=int, default=1, help="Debug level.")
    parser.add_argument("--debug_dir", type=str, default="", help="Debug directory.")
    parser.add_argument("--depth_scale", type=float, default=0.001, help="Depth scale.")
    parser.add_argument("--color_topic", type=str, default="/tiago_head_camera_down/color/image_raw/compressed", help="Color topic.")
    parser.add_argument("--depth_topic", type=str, default="/tiago_head_camera_down/depth/image_raw/compressedDepth", help="Depth topic.")
    parser.add_argument("--camera_info_topic", type=str, default="/tiago_head_camera_down/color/camera_info", help="Camera info topic.")
    parser.add_argument("--pose_frame_id", type=str, default="tiago_head_camera_down_color_optical_frame", help="Pose frame id.")
    parser.add_argument("--slop", type=float, default=1.0, help="Slop.")
    parser.add_argument("--seg_model_name", type=str, default="yolo26n-seg.pt", help="Segmentation model name.")
    parser.add_argument("--resize_factor", type=int, default=1, help="Resize factor to divide the image size by this factor.")
    parser.add_argument("--min_initial_detection_counter", type=int, default=5, help="Minimum initial detection counter.")
    parser.add_argument("--enable_pose_tracking", action="store_true", default=False, help="Enable pose tracking.")
    args = parser.parse_args()
    main(args)

        
