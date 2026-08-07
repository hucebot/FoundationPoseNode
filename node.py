"""
FoundationPose ROS2 node: subscribes to compressed RGB and depth images,
runs open-vocabulary object segmentation (YOLOE, SAM3, or Qwen+SAM2) and 6-DoF pose estimation (register + track).
Use --no_fp to skip pose estimation at runtime (detection/segmentation only; models still load).
Qwen+SAM2 modes: qwen_sam2_detect (bbox prompt) and qwen_sam2_point (point prompt); with --publish_mask_image draw the prompt on the mask image.

Requires: ROS2 (rclpy), sensor_msgs, geometry_msgs, message_filters.
Run from workspace: python run_demo_ros2.py
  (or: ros2 run <your_pkg> run_demo_ros2.py if installed as a package)

Subscriptions:
  - /camera/color/image_raw/compressed (sensor_msgs/CompressedImage)
  - /camera/depth/image_raw/compressed (sensor_msgs/CompressedImage)  [depth_source=realsense]
  - /camera/color/camera_info (sensor_msgs/CameraInfo) for intrinsics K
  - /orchestrator/pose/toggle_fp (std_msgs/Bool) to enable/disable the node
  - /orchestrator/pose/toggle_tracking (std_msgs/Bool) to enable/disable pose tracking
  - /orchestrator/pose/target_object (std_msgs/String) to set the target object text prompt at runtime

Depth source:
  - realsense (default): synchronized RGB + RealSense depth
  - moge: RGB only; metric depth from MoGe-2 (local ./moge + checkpoint)

Publishes:
  - /foundation_pose/object_pose (geometry_msgs/PoseStamped)
  - /foundation_pose/object_marker (visualization_msgs/Marker) mesh marker at pose
  - /foundation_pose/mask_image (sensor_msgs/Image) optional debug mask overlay
  - /foundation_pose/moge_depth/image_raw (sensor_msgs/Image 16UC1 mm) optional MoGe depth
"""

import argparse
import json
import math
import os
import re
import sys
import time
import threading
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker
from message_filters import Subscriber, ApproximateTimeSynchronizer
from PIL import Image as PILImage

from foundationpose.estimater import *
from sensor_msgs.msg import Image as RosImage  # after estimater * (PIL.Image otherwise shadows)
from ultralytics import YOLOE
from ultralytics.models.sam import SAM2Predictor, SAM3SemanticPredictor

from scipy.spatial.transform import Rotation


def _fov_x_deg_from_K(K: np.ndarray, width: int) -> float:
    """Horizontal FOV in degrees from intrinsics fx and image width."""
    fx = float(K[0, 0])
    return float(2.0 * math.degrees(math.atan(0.5 * width / fx)))


def estimate_depth_moge(model, color_rgb: np.ndarray, fov_x: Optional[float] = None, resolution_level: int = 9) -> np.ndarray:
    """Run MoGe-2 on an RGB uint8 image; return metric depth (H, W) float32 with invalids as 0."""
    device = next(model.parameters()).device
    image = torch.tensor(color_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    output = model.infer(image, fov_x=fov_x, resolution_level=resolution_level)
    depth = output["depth"].detach().float().cpu().numpy()
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


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


def _biggest_mask_index(masks):
    """Return the index of the mask with the most foreground pixels."""
    return int(np.argmax([(m > 0).sum() for m in masks]))


def _parse_qwen_bboxes_norm(text: str):
    """Parse Qwen grounding JSON / bare lists; return list of [x1,y1,x2,y2] in 0-1000 coords."""
    if not text:
        return []
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    candidates = []
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            # Bare [x1,y1,x2,y2]
            if len(data) == 4 and all(isinstance(v, (int, float)) for v in data):
                candidates.append([float(v) for v in data])
            else:
                for item in data:
                    if isinstance(item, dict) and "bbox_2d" in item:
                        bbox = item["bbox_2d"]
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            candidates.append([float(v) for v in bbox])
                    elif isinstance(item, (list, tuple)) and len(item) == 4:
                        candidates.append([float(v) for v in item])
    except Exception:
        pass

    if not candidates:
        for match in re.finditer(
            r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]',
            text,
        ):
            candidates.append([float(match.group(i)) for i in range(1, 5)])
    if not candidates:
        for match in re.finditer(
            r'\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]',
            text,
        ):
            candidates.append([float(match.group(i)) for i in range(1, 5)])
    return candidates


def _parse_qwen_points_norm(text: str):
    """Parse Qwen grounding JSON / bare lists; return list of [x,y] in 0-1000 coords."""
    if not text:
        return []
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    candidates = []
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            # Bare [x,y]
            if len(data) == 2 and all(isinstance(v, (int, float)) for v in data):
                candidates.append([float(v) for v in data])
            else:
                for item in data:
                    if isinstance(item, dict) and "point_2d" in item:
                        pt = item["point_2d"]
                        if isinstance(pt, (list, tuple)) and len(pt) == 2:
                            candidates.append([float(v) for v in pt])
                    elif isinstance(item, (list, tuple)) and len(item) == 2:
                        candidates.append([float(v) for v in item])
    except Exception:
        pass

    if not candidates:
        for match in re.finditer(
            r'"point_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]',
            text,
        ):
            candidates.append([float(match.group(1)), float(match.group(2))])
    if not candidates:
        for match in re.finditer(
            r'\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]',
            text,
        ):
            candidates.append([float(match.group(1)), float(match.group(2))])
    return candidates


def _norm1000_bbox_to_xyxy(bbox_norm, width: int, height: int):
    """Convert a Qwen 0-1000 bbox to clipped integer pixel xyxy."""
    x1 = int(round(bbox_norm[0] / 1000.0 * width))
    y1 = int(round(bbox_norm[1] / 1000.0 * height))
    x2 = int(round(bbox_norm[2] / 1000.0 * width))
    y2 = int(round(bbox_norm[3] / 1000.0 * height))
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _norm1000_point_to_xy(point_norm, width: int, height: int):
    """Convert a Qwen 0-1000 point to clipped integer pixel xy."""
    x = int(round(point_norm[0] / 1000.0 * width))
    y = int(round(point_norm[1] / 1000.0 * height))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return [x, y]


def _qwen_prompt_for_mode(vlm_prompt: str, seg_model_type: str):
    """Detect mode keeps Detect; point mode swaps Detect -> Locate."""
    if seg_model_type == "qwen_sam2_point":
        return vlm_prompt.replace("Detect", "Locate")
    return vlm_prompt


def _qwen_special_token_id(processor, token_str: str, fallback: int):
    """Resolve a special token id from the processor/tokenizer, else use fallback."""
    tok = getattr(processor, "tokenizer", processor)
    convert = getattr(tok, "convert_tokens_to_ids", None)
    if convert is None:
        return fallback
    tid = convert(token_str)
    unk = getattr(tok, "unk_token_id", None)
    if tid is None or tid < 0 or (unk is not None and tid == unk):
        return fallback
    return int(tid)


def _qwen_generate_kwargs():
    """Shared sampling kwargs for Qwen generate calls."""
    return dict(
        do_sample=True,
        temperature=0.7,
        top_p=0.80,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
    )


def _qwen_generate(
    processor,
    model,
    color_rgb: np.ndarray,
    full_prompt: str,
    max_new_tokens: int = 128,
    thinking_budget: int = 256,
    logger=None,
):
    """Run Qwen VLM; return (answer_text, thinking_text).

    Thinking budget is applied as a single-shot max_new_tokens cap
    (thinking_budget + answer tokens). Two-stage early-stop continuation
    is unsafe for Qwen3.5 multimodal (rope/mask shape mismatch).
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": PILImage.fromarray(color_rgb)},
                {"type": "text", "text": full_prompt},
            ],
        }
    ]
    # ProcessorMixin expects conversation as the first positional arg (not messages=)
    chat_kwargs = dict(
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    try:
        inputs = processor.apply_chat_template(messages, enable_thinking=True, **chat_kwargs)
    except TypeError:
        inputs = processor.apply_chat_template(messages, **chat_kwargs)
    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    sample_kwargs = _qwen_generate_kwargs()
    total_max_new_tokens = max(thinking_budget, 0) + max(max_new_tokens, 1)
    think_end_id = _qwen_special_token_id(processor, "</think>", 151668)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=total_max_new_tokens,
            **sample_kwargs,
        )
    output_ids = generated_ids[0][input_len:].tolist()

    # Prefer the final answer after the last </think>
    try:
        answer_start = len(output_ids) - output_ids[::-1].index(think_end_id)
    except ValueError:
        answer_start = 0
    thinking_text = ""
    if answer_start > 0:
        thinking_text = processor.decode(output_ids[:answer_start], skip_special_tokens=True).strip()
        # Model sometimes emits a literal "</think>" string inside the thinking span
        thinking_text = thinking_text.replace("</think>", "").strip()
    answer_text = processor.decode(output_ids[answer_start:], skip_special_tokens=True).strip()
    return answer_text, thinking_text


def qwen_predict_bbox_xyxy(
    processor,
    model,
    color_rgb: np.ndarray,
    vlm_prompt: str,
    max_new_tokens: int = 128,
    thinking_budget: int = 256,
    logger=None,
):
    """Ask Qwen for a bbox; return (xyxy_pixels or None, raw_text)."""
    h, w = color_rgb.shape[:2]
    full_prompt = (
        f"{vlm_prompt}. "
        "Return their locations in the form of coordinates in the format "
        "{'bbox_2d': [x1, y1, x2, y2], label: 'object_name'}."
        "Use normalized coordinates in the range [0, 1000] and return only a JSON string."
    )
    raw_text, thinking_text = _qwen_generate(
        processor,
        model,
        color_rgb,
        full_prompt,
        max_new_tokens=max_new_tokens,
        thinking_budget=thinking_budget,
        logger=logger,
    )
    if logger is not None:
        if thinking_text:
            logger.info(f"Qwen thinking: {thinking_text}")
        logger.info(f"Qwen raw text: {raw_text}")
    bboxes_norm = _parse_qwen_bboxes_norm(raw_text)
    if not bboxes_norm and thinking_text:
        bboxes_norm = _parse_qwen_bboxes_norm(thinking_text)
    if not bboxes_norm:
        return None, raw_text

    # Prefer the largest box when several are returned
    xyxy_list = []
    for bbox_norm in bboxes_norm:
        xyxy = _norm1000_bbox_to_xyxy(bbox_norm, w, h)
        if xyxy is not None:
            xyxy_list.append(xyxy)
    if not xyxy_list:
        return None, raw_text
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in xyxy_list]
    return xyxy_list[int(np.argmax(areas))], raw_text


def qwen_predict_point_xy(
    processor,
    model,
    color_rgb: np.ndarray,
    vlm_prompt: str,
    max_new_tokens: int = 128,
    thinking_budget: int = 256,
    logger=None,
):
    """Ask Qwen for a single point; return (xy_pixels or None, raw_text)."""
    h, w = color_rgb.shape[:2]
    full_prompt = (
        f"{vlm_prompt}. "
        "Return their locations as a single point in the form of coordinates in the format "
        "{'point_2d': [x, y], label: 'object_name'}."
        "Use normalized coordinates in the range [0, 1000] and return only a JSON string."
    )
    raw_text, thinking_text = _qwen_generate(
        processor,
        model,
        color_rgb,
        full_prompt,
        max_new_tokens=max_new_tokens,
        thinking_budget=thinking_budget,
        logger=logger,
    )
    if logger is not None:
        if thinking_text:
            logger.info(f"Qwen thinking: {thinking_text}")
        logger.info(f"Qwen raw text: {raw_text}")
    points_norm = _parse_qwen_points_norm(raw_text)
    if not points_norm and thinking_text:
        points_norm = _parse_qwen_points_norm(thinking_text)
    if not points_norm:
        return None, raw_text
    # Prefer the first valid point (prompt asks for a single location)
    return _norm1000_point_to_xy(points_norm[0], w, h), raw_text


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
                  "vlm_prompt": "Detect the bread baguette",
                  "constraint_yaw_in": 0,
                  "constraint_pitch_in": [0,180],
                  "constraint_roll_in": [0,180]},

    # seems okay and does not flip
    "banana" : {"mesh_file": "./assets/hackathon3/banana/banana.obj",
                "symmetry_x_angles": "",
                "symmetry_y_angles": "",
                "symmetry_z_angles": "",
                "target_object": "banana",
                "vlm_prompt": "Detect the banana",
                "constraint_yaw_in": None,
                "constraint_pitch_in": None,
                "constraint_roll_in": None},

    # still flips up and down, adjusted size to retry
    "coffeecan" : {"mesh_file": "./assets/hackathon3/coffeecan/coffeecan.obj",
                   "symmetry_x_angles": "",
                   "symmetry_y_angles": "",
                   "symmetry_z_angles": "",
                   "target_object": "blue container",
                   "vlm_prompt": "Detect the blue container with yellow cap",
                   "constraint_yaw_in": 0,
                   "constraint_pitch_in": None,
                   "constraint_roll_in": None},

    # lots of flipping difficult to stabilize
    "egg" : {"mesh_file": "./assets/hackathon3/egg/egg.obj",
             "symmetry_x_angles": "0,180",
             "symmetry_y_angles": "0,180",
             "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
             "target_object": "egg",
             "vlm_prompt": "Detect the egg",
             "constraint_yaw_in": 0,
             "constraint_pitch_in": 0,
             "constraint_roll_in": 0},

    # good it seems that does not flip up/down but the handle does not always get oriented correctly
    "flowercup" : {"mesh_file": "./assets/hackathon3/flowercup/flowercup.obj",
                   "symmetry_x_angles": "",
                   "symmetry_y_angles": "",
                   "symmetry_z_angles": "",
                   "target_object": "yellow mug",
                   "vlm_prompt": "Detect the yellow mug",
                   "constraint_yaw_in": None,
                   "constraint_pitch_in": None,
                   "constraint_roll_in": None},

    # not detected with keyword jam
    "jam" : {"mesh_file": "./assets/hackathon3/jam/jam.obj",
             "symmetry_x_angles": "",
             "symmetry_y_angles": "",
             "symmetry_z_angles": "",
             "target_object": "orange jam",
             "vlm_prompt": "Detect the orange jam pot",
             "constraint_yaw_in": None,
             "constraint_pitch_in": None,
             "constraint_roll_in": None},

    # ok
    "milk" : {"mesh_file": "./assets/hackathon3/milk/milk.obj",
              "symmetry_x_angles": "",
              "symmetry_y_angles": "",
              "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
              "target_object": "white bottle",
              "vlm_prompt": "Detect the white bottle with a blue cap and a blue label",
              "constraint_yaw_in": 0,
              "constraint_pitch_in": None,
              "constraint_roll_in": None},

    # hard to find but seems good when found... investigate, maybe a bigger one !
    "minicheese" : {"mesh_file": "./assets/hackathon3/minicheese/minicheese.obj",
                    "symmetry_x_angles": "",
                    "symmetry_y_angles": "",
                    "symmetry_z_angles": "",
                    "target_object": "triangle cheese",
                    "vlm_prompt": "Detect the triangle cheese block",
                    "constraint_yaw_in": None,
                    "constraint_pitch_in": None,
                    "constraint_roll_in": None},

    # seems robust could be a good anchor point
    "pan" : {"mesh_file": "./assets/hackathon3/pan/pan.obj",
             "symmetry_x_angles": "",
             "symmetry_y_angles": "",
             "symmetry_z_angles": "",
             "target_object": "pan",
             "vlm_prompt": "Detect the pan",
             "constraint_yaw_in": None,
             "constraint_pitch_in": None,
             "constraint_roll_in": None},

    # problem gets a lot of multiple objects + rotations
    "redapple" : {"mesh_file": "./assets/hackathon3/redapple/redapple.obj",
                  "symmetry_x_angles": "0,180",
                  "symmetry_y_angles": "0,180",
                  "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
                  "target_object": "red apple",
                  "vlm_prompt": "Detect the red apple",
                  "constraint_yaw_in": 0,
                  "constraint_pitch_in": 0,
                  "constraint_roll_in": 0},

    "smallmilk" : {"mesh_file": "./assets/hackathon3/smallmilk/smallmilk.obj",
                   "symmetry_x_angles": "",
                   "symmetry_y_angles": "",
                   "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
                   "target_object": "white bottle",
                   "vlm_prompt": "Detect the small white bottle with a blue cap",
                   "constraint_yaw_in": 0,
                   "constraint_pitch_in": None,
                   "constraint_roll_in": None},

    "smallsanpellegrino" : {"mesh_file": "./assets/hackathon3/smallsanpellegrino/smallsanpellegrino.obj",
                            "symmetry_x_angles": "",
                            "symmetry_y_angles": "",
                            "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
                            "target_object": "green bottle",
                            "vlm_prompt": "Detect the small green bottle",
                            "constraint_yaw_in": 0,
                            "constraint_pitch_in": None,
                            "constraint_roll_in": None},

    # hard to detect and flips on several axes
    "spam" : {"mesh_file": "./assets/hackathon3/spam/spam.obj",
              "symmetry_x_angles": "",
              "symmetry_y_angles": "",
              "symmetry_z_angles": "",
              "target_object": "blue container",
              "vlm_prompt": "Detect the SPAM blue container",
              "constraint_yaw_in": [0,180],
              "constraint_pitch_in": [0,180],
              "constraint_roll_in": [0,180]},

    # okay but flips on one axis
    "ycbmustard" : {"mesh_file": "./assets/hackathon3/ycbmustard/ycbmustard.obj",
                    "symmetry_x_angles": "",
                    "symmetry_y_angles": "",
                    "symmetry_z_angles": "0,180",
                    "target_object": "yellow bottle",
                    "vlm_prompt": "Detect the yellow mustard bottle",
                    "constraint_yaw_in": [0,180],
                    "constraint_pitch_in": None,
                    "constraint_roll_in": None},

    "juice" : { "mesh_file": "./assets/hackathon3/juice/juice.obj",
                "symmetry_x_angles": "",
                "symmetry_y_angles": "",
                "symmetry_z_angles": "0,90,180,270",
                "target_object": "carton bottle",
                "vlm_prompt": "Detect the red and white carton bottle",
                "constraint_yaw_in": [0,90],
                "constraint_pitch_in": None,
                "constraint_roll_in": None},

    "redcup" : {"mesh_file": "./assets/hackathon3/redcup/redcup.obj",
                   "symmetry_x_angles": "",
                   "symmetry_y_angles": "",
                   "symmetry_z_angles": "",
                   "target_object": "red mug",
                   "vlm_prompt": "Detect the red and white mug",
                   "constraint_yaw_in": None,
                   "constraint_pitch_in": None,
                   "constraint_roll_in": None},

    "basket" : {"mesh_file": "./assets/hackathon3/basket/basket.obj",
                   "symmetry_x_angles": "",
                   "symmetry_y_angles": "",
                   "symmetry_z_angles": "",
                   "target_object": "fruit basket",
                   "vlm_prompt": "Detect the fruit basket",
                   "constraint_yaw_in": None,
                   "constraint_pitch_in": None,
                   "constraint_roll_in": None},

    "perrier" : {"mesh_file": "./assets/hackathon3/perrier/perrier.obj",
              "symmetry_x_angles": "",
              "symmetry_y_angles": "",
              "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
              "target_object": "green bottle",
              "vlm_prompt": "Detect the big green bottle",
              "constraint_yaw_in": 0,
              "constraint_pitch_in": None,
              "constraint_roll_in": None},

    "solevita" : {"mesh_file": "./assets/hackathon3/solevita/solevita.obj",
              "symmetry_x_angles": "",
              "symmetry_y_angles": "",
              "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
              "target_object": "green bottle",
              "vlm_prompt": "Detect the small 'Solevita' green and white bottle with a yellow cap",
              "constraint_yaw_in": 0,
              "constraint_pitch_in": None,
              "constraint_roll_in": None},

    "multifruit" : { "mesh_file": "./assets/hackathon3/multifruit/multifruit.obj",
                "symmetry_x_angles": "0,180",
                "symmetry_y_angles": "0,180",
                "symmetry_z_angles": "0,90,180,270",
                "target_object": "carton bottle",
                "vlm_prompt": "Detect the juice carton bottle with the green top and white cap",
                "constraint_yaw_in": [0,90],
                "constraint_pitch_in": [0,180],
                "constraint_roll_in": [0,180]},

    "blackcup" : { "mesh_file": "./assets/hackathon3/blackcup/blackcup.obj",
                "symmetry_x_angles": "",
                "symmetry_y_angles": "",
                "symmetry_z_angles": "",
                "target_object": "black mug",
                "vlm_prompt": "Detect the black cup",
                "constraint_yaw_in": None,
                "constraint_pitch_in": None,
                "constraint_roll_in": None},

    "thermos" : { "mesh_file": "./assets/hackathon3/thermos/thermos.obj",
                "symmetry_x_angles": "",
                "symmetry_y_angles": "",
                "symmetry_z_angles": "",
                "target_object": "orange mug",
                "vlm_prompt": "Detect the orange cup with a white lid",
                "constraint_yaw_in": None,
                "constraint_pitch_in": None,
                "constraint_roll_in": None},

    "gavottes" : { "mesh_file": "./assets/hackathon3/gavottes/gavottes.obj",
                "symmetry_x_angles": "0,180",
                "symmetry_y_angles": "0,180",
                "symmetry_z_angles": "0,180",
                "target_object": "carton",
                "vlm_prompt": "Detect the white and blue biscuit carton",
                "constraint_yaw_in": [0,180],
                "constraint_pitch_in": [0,180],
                "constraint_roll_in": [0,180]},

    "orange" : {"mesh_file": "./assets/hackathon3/orange/orange.obj",
                  "symmetry_x_angles": "0,180",
                  "symmetry_y_angles": "0,180",
                  "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
                  "target_object": "orange",
                  "vlm_prompt": "Detect the orange",
                  "constraint_yaw_in": 0,
                  "constraint_pitch_in": 0,
                  "constraint_roll_in": 0},

    "bluecup" : {"mesh_file": "./assets/hackathon3/bluecup/bluecup.obj",
              "symmetry_x_angles": "",
              "symmetry_y_angles": "",
              "symmetry_z_angles": "0,30,60,90,120,150,180,210,240,270,300,330",
              "target_object": "blue cup",
              "vlm_prompt": "Detect the blue cup with a gray lid",
              "constraint_yaw_in": 0,
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
        self.declare_parameter("depth_source", args.depth_source)
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
        self.depth_source = self.get_parameter("depth_source").value
        self.pose_frame_id = self.get_parameter("pose_frame_id").value
        self.slop = self.get_parameter("slop").value
        self.seg_model_name = args.seg_model_name
        self.resize_factor = args.resize_factor
        self.min_initial_detection_counter = args.min_initial_detection_counter
        self.enable_pose_tracking = args.enable_pose_tracking
        self.no_fp = args.no_fp
        self.publish_mask_image = args.publish_mask_image
        self.publish_moge_depth = args.publish_moge_depth
        self.moge_depth_topic = args.moge_depth_topic
        self.seg_model_type = args.seg_model_type
        self.yoloe_conf = args.yoloe_conf
        self.qwen_model_name = args.qwen_model
        self.qwen_thinking_budget = args.qwen_thinking_budget
        self.vlm_prompt = args.vlm_prompt
        self.multiple_object_method = args.multiple_object_method
        self.symmetry_x_angles = args.symmetry_x_angles
        self.symmetry_y_angles = args.symmetry_y_angles
        self.symmetry_z_angles = args.symmetry_z_angles
        self.fp_verbosity = args.fp_verbosity
        self.use_onnx = args.use_onnx
        self.refiner_onnx = args.refiner_onnx
        self.scorer_onnx = args.scorer_onnx
        self.prefer_tensorrt = not args.no_prefer_tensorrt
        self.moge_checkpoint = args.moge_checkpoint
        self.moge_resolution_level = args.moge_resolution_level
        self.moge_use_camera_fov = args.moge_use_camera_fov
        self.moge_model = None
        if self.fp_verbosity not in ["debug", "info", "warning", "error", "critical"]:
            raise ValueError(f"Invalid verbosity: {self.fp_verbosity}. Valid: debug, info, warning, error, critical")
        if self.multiple_object_method not in ["none", "biggest"]:
            raise ValueError(f"Invalid multiple_object_method: {self.multiple_object_method}. Valid: none, biggest")
        if self.depth_source not in ["realsense", "moge"]:
            raise ValueError(f"Invalid depth_source: {self.depth_source}. Valid: realsense, moge")
        if self.publish_moge_depth and self.depth_source != "moge":
            raise ValueError("--publish_moge_depth requires --depth_source moge")

        # Make some checks on the parameters
        assert(self.seg_model_type in ["sam3", "yoloe", "qwen_sam2_detect", "qwen_sam2_point"]), f"Invalid segmentation model type: {self.seg_model_type}"
        if self.seg_model_type == "sam3":
            self.seg_model_name = "sam3.pt"
        elif self.seg_model_type in ("qwen_sam2_detect", "qwen_sam2_point"):
            if "sam2" not in self.seg_model_name:
                self.seg_model_name = "sam2_s.pt"
        elif self.seg_model_type == "yoloe":
            assert("yoloe" in self.seg_model_name), f"Invalid YOLOE model name: {self.seg_model_name}"

        # Print parameters
        self.get_logger().debug("==== PARAMETERS ====")
        self.get_logger().debug(f"Object key: {self.object_key}")
        self.get_logger().debug(f"Mesh file: {self.mesh_file}")
        self.get_logger().debug(f"Target object: {self.target_object}")
        self.get_logger().debug(f"VLM prompt: {self.vlm_prompt}")
        self.get_logger().debug(f"Est refine iter: {self.est_refine_iter}")
        self.get_logger().debug(f"Track refine iter: {self.track_refine_iter}")
        self.get_logger().debug(f"Debug: {self.debug}")
        self.get_logger().debug(f"Debug dir: {self.debug_dir}")
        self.get_logger().debug(f"Depth scale: {self.depth_scale}")
        self.get_logger().debug(f"Depth source: {self.depth_source}")
        self.get_logger().debug(f"Pose frame id: {self.pose_frame_id}")
        self.get_logger().debug(f"Slop: {self.slop}")
        self.get_logger().debug(f"Resize factor: {self.resize_factor}")
        self.get_logger().debug(f"Min initial detection counter: {self.min_initial_detection_counter}")
        self.get_logger().debug(f"YOLOE confidence threshold: {self.yoloe_conf}")
        self.get_logger().debug(f"Qwen model: {self.qwen_model_name}")
        self.get_logger().debug(f"Qwen thinking budget: {self.qwen_thinking_budget}")
        self.get_logger().debug(f"Multiple object method: {self.multiple_object_method}")
        self.get_logger().debug(f"Enable pose tracking: {self.enable_pose_tracking}")
        self.get_logger().debug(f"No FoundationPose (--no_fp): {self.no_fp}")
        self.get_logger().debug(f"Publish mask image: {self.publish_mask_image}")
        self.get_logger().debug(f"Publish MoGe depth: {self.publish_moge_depth}")
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
        if self.depth_source == "moge":
            self.get_logger().debug(f"MoGe checkpoint: {self.moge_checkpoint}")
            self.get_logger().debug(f"MoGe resolution level: {self.moge_resolution_level}")
            self.get_logger().debug(f"MoGe use camera FOV: {self.moge_use_camera_fov}")

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
        self.vlm_processor = None
        self.vlm_model = None
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
        elif self.seg_model_type in ("qwen_sam2_detect", "qwen_sam2_point"):
            # Qwen locates a bbox or point; SAM2 prompt turns that into a mask (lighter than SAM3)
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            self.get_logger().info(f"Loading Qwen VLM {self.qwen_model_name}...")
            self.vlm_processor = AutoProcessor.from_pretrained(self.qwen_model_name)
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            self.vlm_model = AutoModelForMultimodalLM.from_pretrained(
                self.qwen_model_name,
                torch_dtype=dtype,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.vlm_model.eval()
            self.get_logger().info(f"Qwen VLM ready (dtype={dtype}, mode={self.seg_model_type})")

            overrides = dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=self.seg_model_name,
                half=True,
                save=False,
                verbose=False,
            )
            self.seg_model = SAM2Predictor(overrides=overrides)
            self.seg_model.set_image(np.zeros((1080, 1920, 3), dtype=np.uint8))
            if self.seg_model_type == "qwen_sam2_point":
                self.seg_model(points=[[200, 250]], labels=[1])
            else:
                self.seg_model(bboxes=[[100, 100, 300, 400]])
        elif self.seg_model_type == "yoloe":
            # open vocabulary: the target object is given as a text prompt, no fixed class list
            self.seg_model = YOLOE(self.seg_model_name)
            self.seg_model.set_classes([self.target_object])
            # run a fake pass to warm up the model
            self.seg_model.predict(np.zeros((1080, 1920, 3), dtype=np.uint8), conf=self.yoloe_conf, verbose=False)
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

        # Optional MoGe monocular depth (replaces RealSense depth)
        if self.depth_source == "moge":
            moge_root = os.path.join(code_dir, "moge")
            if moge_root not in sys.path:
                sys.path.insert(0, moge_root)
            if not os.path.isfile(self.moge_checkpoint):
                raise FileNotFoundError(f"MoGe checkpoint not found: {self.moge_checkpoint}")
            self.get_logger().info(f"Loading MoGe-2 from {self.moge_checkpoint}...")
            from moge.model.v2 import MoGeModel
            self.moge_model = MoGeModel.from_pretrained(self.moge_checkpoint).cuda().eval()
            self.get_logger().info(
                f"MoGe-2 ready (resolution_level={self.moge_resolution_level}, use_camera_fov={self.moge_use_camera_fov})"
            )

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

        self._moge_depth_pub = None
        if self.publish_moge_depth:
            self._moge_depth_pub = self.create_publisher(
                RosImage,
                self.moge_depth_topic,
                qos_sensor,
            )
            self.get_logger().info(f"Publishing MoGe depth (16UC1 mm) on {self.moge_depth_topic}")

        self._toggle_fp_sub = self.create_subscription(
            Bool,
            "/orchestrator/foundation_pose/toggle",
            self._toggle_fp_cb,
            qos_state,
        )

        self._toggle_tracking_sub = self.create_subscription(
            Bool,
            "/orchestrator/foundation_pose/toggle_tracking",
            self._toggle_tracking_cb,
            qos_state,
        )

        self._target_object_sub = self.create_subscription(
            String,
            "/orchestrator/foundation_pose/target_object",
            self._target_object_cb,
            qos_cmd,
        )

        if self.depth_source == "realsense":
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
        else:
            # Color-only: MoGe predicts metric depth from RGB
            self._color_sub = self.create_subscription(
                CompressedImage,
                self.get_parameter("color_topic").value,
                self._color_cb,
                qos_sensor,
            )
            self.get_logger().info(
                f"Subscribed to {self.get_parameter('color_topic').value} (depth_source=moge); waiting for camera_info and color messages"
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
            self.vlm_prompt = OBJECT_KEYS_TO_PARAMETERS[key_name]["vlm_prompt"]
            if self.seg_model_type == "yoloe":
                self.seg_model.set_classes([self.target_object])
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
            # just a change of target object, any text prompt is valid with an open vocabulary model
            self.target_object = new_target
            self.vlm_prompt = f"Detect the {self.target_object}"
            if self.seg_model_type == "yoloe":
                self.seg_model.set_classes([self.target_object])
            self.get_logger().info(f"Target object changed to: {self.target_object} (vlm_prompt={self.vlm_prompt})")

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

    def _publish_moge_depth_image(self, depth_m: np.ndarray, header, out_hw=None):
        """Publish MoGe metric depth as RealSense-compatible 16UC1 image_raw (millimeters)."""
        if self._moge_depth_pub is None:
            return
        depth = depth_m
        if out_hw is not None and (depth.shape[0] != out_hw[0] or depth.shape[1] != out_hw[1]):
            depth = cv2.resize(depth, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
        # meters -> uint16 mm (same convention as RealSense + depth_scale=0.001)
        depth_u16 = np.clip(np.round(depth / self.depth_scale), 0, 65535).astype(np.uint16)
        depth_u16 = np.ascontiguousarray(depth_u16)
        msg = RosImage()
        msg.header.stamp = header.stamp
        msg.header.frame_id = self.pose_frame_id
        msg.height = int(depth_u16.shape[0])
        msg.width = int(depth_u16.shape[1])
        msg.encoding = "16UC1"
        msg.is_bigendian = 0
        msg.step = int(depth_u16.shape[1] * 2)
        msg.data = depth_u16.tobytes()
        self._moge_depth_pub.publish(msg)

    def _publish_mask_image(self, rgb, masks, header, bboxes=None, points=None):
        """Overlay masks and/or xyxy bboxes / points on rgb (HxWx3) and publish as raw Image."""
        mask_list = [] if masks is None else list(masks)
        bbox_list = [] if bboxes is None else [b for b in bboxes if b is not None and len(b) == 4]
        point_list = [] if points is None else [p for p in points if p is not None and len(p) == 2]
        if self._mask_image_pub is None or (len(mask_list) == 0 and len(bbox_list) == 0 and len(point_list) == 0):
            return
        t0 = time.time()
        vis = rgb.copy()
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        ]
        for i, mask in enumerate(mask_list):
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

        for i, bbox in enumerate(bbox_list):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            box_color = colors[i % len(colors)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

        for i, pt in enumerate(point_list):
            x, y = int(pt[0]), int(pt[1])
            pt_color = colors[i % len(colors)]
            cv2.circle(vis, (x, y), 8, pt_color, 2)
            cv2.drawMarker(vis, (x, y), pt_color, markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)

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
        self.get_logger().debug(
            f"Publish mask image: {(time.time()-t0)*1000:.2f} ms "
            f"({len(mask_list)} mask(s), {len(bbox_list)} bbox(es), {len(point_list)} point(s))"
        )

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

    def _color_cb(self, color_msg: CompressedImage):
        """RGB-only path when depth_source=moge."""
        self._rgbd_cb(color_msg, None)

    def _rgbd_cb(self, color_msg: CompressedImage, depth_msg: Optional[CompressedImage]):
        self.get_logger().debug("Received RGBD message" if depth_msg is not None else "Received color message")
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

            if self.depth_source == "moge":
                # Keep full-res size for optional depth publish (match RealSense 16UC1 image_raw)
                publish_hw = color.shape[:2]
                # Run MoGe on the resolution FoundationPose will use
                if self.resize_factor != 1:
                    color = cv2.resize(
                        color,
                        (color.shape[1] // self.resize_factor, color.shape[0] // self.resize_factor),
                        interpolation=cv2.INTER_LINEAR,
                    )
                fov_x = _fov_x_deg_from_K(self.K, color.shape[1]) if self.moge_use_camera_fov else None
                t0 = time.time()
                depth = estimate_depth_moge(
                    self.moge_model,
                    color,
                    fov_x=fov_x,
                    resolution_level=self.moge_resolution_level,
                )
                moge_ms = (time.time() - t0) * 1000.0
                self.get_logger().debug(
                    f"MoGe depth {depth.shape} in {moge_ms:.1f} ms (fov_x={fov_x})"
                )
                if self.publish_moge_depth:
                    self._publish_moge_depth_image(depth, color_msg.header, out_hw=publish_hw)
            else:
                if depth_msg is None:
                    raise ValueError("depth_msg is required when depth_source=realsense")
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
        except Exception as e:
            self.get_logger().error(f"Failed to prepare RGBD: {e}")
            self._lock.acquire()
            self._processing = False
            self._lock.release()
            return

        self.rgbd_frames_counter_processed += 1

        valid_pose = False
        if "Tracking" not in self.current_phase:
            self.current_phase = "Detecting"

            if self.seg_model_type == "sam3":
                self.seg_model.set_image(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
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
                    if self.multiple_object_method == "biggest":
                        masks_np = [target_masks[i].data.cpu().numpy()[0] for i in range(found_obects)]
                        best_idx = _biggest_mask_index(masks_np)
                        target_mask = masks_np[best_idx].astype(np.uint8)
                        self.initial_detection_counter = self.min_initial_detection_counter
                        self.get_logger().info(
                            f"Multiple objects ({found_obects}); selected biggest mask idx={best_idx} "
                            f"({int((target_mask > 0).sum())} px)"
                        )
                    else:
                        self.get_logger().warn(f"Multiple objects found ({found_obects}) in frame {self.rgbd_frames_counter_processed}, cannot chose")
                        self.initial_detection_counter = 0
                else:
                    self.initial_detection_counter = 0

            elif self.seg_model_type in ("qwen_sam2_detect", "qwen_sam2_point"):
                # Qwen -> bbox or point, then SAM2 prompt -> mask
                target_mask = None
                try:
                    use_point = self.seg_model_type == "qwen_sam2_point"
                    qwen_prompt = _qwen_prompt_for_mode(self.vlm_prompt, self.seg_model_type)
                    qwen_t0 = time.time()
                    if use_point:
                        prompt_xy, qwen_text = qwen_predict_point_xy(
                            self.vlm_processor,
                            self.vlm_model,
                            color,
                            qwen_prompt,
                            thinking_budget=self.qwen_thinking_budget,
                            logger=self.get_logger(),
                        )
                    else:
                        prompt_xy, qwen_text = qwen_predict_bbox_xyxy(
                            self.vlm_processor,
                            self.vlm_model,
                            color,
                            qwen_prompt,
                            thinking_budget=self.qwen_thinking_budget,
                            logger=self.get_logger(),
                        )
                    qwen_ms = (time.time() - qwen_t0) * 1000.0
                    prompt_kind = "point" if use_point else "bbox"
                    if prompt_xy is None:
                        self.get_logger().warn(
                            f"Frame {self.rgbd_frames_counter_processed}: Qwen found no {prompt_kind} "
                            f"(prompt={qwen_prompt!r}, {qwen_ms:.1f} ms). Raw: {qwen_text!r}"
                        )
                        self.initial_detection_counter = 0
                    else:
                        self.get_logger().info(
                            f"Frame {self.rgbd_frames_counter_processed}: Qwen {prompt_kind}={prompt_xy} "
                            f"({qwen_ms:.1f} ms, prompt={qwen_prompt!r})"
                        )
                        sam_t0 = time.time()
                        self.seg_model.set_image(color)
                        if use_point:
                            results = self.seg_model(points=[prompt_xy], labels=[1])
                        else:
                            results = self.seg_model(bboxes=[prompt_xy])
                        sam_ms = (time.time() - sam_t0) * 1000.0
                        n_masks = len(results[0].masks) if results is not None and results[0].masks is not None else 0
                        sam_prompt = "point" if use_point else "box"
                        self.get_logger().info(
                            f"Frame {self.rgbd_frames_counter_processed}: SAM2({sam_prompt}) found {n_masks} mask(s), "
                            f"latency: {sam_ms:.2f} ms (Qwen+SAM2 total {(qwen_ms + sam_ms):.1f} ms)"
                        )
                        pub_bboxes = None if use_point else [prompt_xy]
                        pub_points = [prompt_xy] if use_point else None
                        if results is None or results[0].masks is None or n_masks == 0:
                            self.get_logger().error(
                                f"No SAM2 mask from Qwen {prompt_kind} for frame {self.rgbd_frames_counter_processed}"
                            )
                            if self.publish_mask_image:
                                self._publish_mask_image(
                                    color, [], color_msg.header, bboxes=pub_bboxes, points=pub_points
                                )
                            self.initial_detection_counter = 0
                        else:
                            target_masks = results[0].masks
                            found_obects = len(target_masks)
                            if self.publish_mask_image:
                                masks_list = [target_masks[i].data.cpu().numpy()[0] for i in range(found_obects)]
                                self._publish_mask_image(
                                    color, masks_list, color_msg.header, bboxes=pub_bboxes, points=pub_points
                                )
                            if found_obects == 1:
                                self.initial_detection_counter = self.min_initial_detection_counter
                                target_mask = target_masks[0].data.cpu().numpy()[0].astype(np.uint8)
                            elif found_obects > 1:
                                if self.multiple_object_method == "biggest":
                                    masks_np = [target_masks[i].data.cpu().numpy()[0] for i in range(found_obects)]
                                    best_idx = _biggest_mask_index(masks_np)
                                    target_mask = masks_np[best_idx].astype(np.uint8)
                                    self.initial_detection_counter = self.min_initial_detection_counter
                                    self.get_logger().info(
                                        f"Multiple SAM2 masks ({found_obects}); selected biggest idx={best_idx} "
                                        f"({int((target_mask > 0).sum())} px)"
                                    )
                                else:
                                    self.get_logger().warn(
                                        f"Multiple SAM2 masks ({found_obects}) in frame "
                                        f"{self.rgbd_frames_counter_processed}, cannot chose"
                                    )
                                    self.initial_detection_counter = 0
                            else:
                                self.initial_detection_counter = 0
                except Exception as e:
                    self.get_logger().error(
                        f"Frame {self.rgbd_frames_counter_processed}: Qwen/SAM2 failed, skipping frame: {e}"
                    )
                    self.initial_detection_counter = 0
                    target_mask = None

            elif self.seg_model_type == "yoloe":
                # the model is already prompted with self.target_object, so every returned mask is a candidate
                target_mask = None
                results = self.seg_model.predict(cv2.cvtColor(color, cv2.COLOR_RGB2BGR), conf=self.yoloe_conf, verbose=False) # ultralytics expects BGR
                masks = results[0].masks
                target_masks_list = [] if masks is None else list(masks.data.cpu().numpy())
                found_object = len(target_masks_list)

                speed = results[0].speed
                total_latency = speed.get("preprocess", 0) + speed.get("inference", 0) + speed.get("postprocess", 0)
                self.get_logger().info(
                    f"Frame {self.rgbd_frames_counter_processed}: YOLOE found {found_object} '{self.target_object}', latency: {total_latency:.2f} ms"
                )
                self.get_logger().debug(f"Scores: {results[0].boxes.conf.cpu().numpy()}")

                if self.publish_mask_image and target_masks_list:
                    self._publish_mask_image(color, target_masks_list, color_msg.header)

                if found_object == 1:
                    # need min_initial_detection_counter detections in a row to start tracking
                    target_mask = target_masks_list[0]
                    self.initial_detection_counter += 1
                    self.get_logger().info(f"Initial detection counter ({self.target_object}): {self.initial_detection_counter} / {self.min_initial_detection_counter}")
                elif found_object > 1:
                    if self.multiple_object_method == "biggest":
                        best_idx = _biggest_mask_index(target_masks_list)
                        target_mask = target_masks_list[best_idx]
                        self.initial_detection_counter += 1
                        self.get_logger().info(
                            f"Multiple objects ({found_object}); selected biggest mask idx={best_idx} "
                            f"({int((target_mask > 0).sum())} px), "
                            f"detection counter: {self.initial_detection_counter} / {self.min_initial_detection_counter}"
                        )
                    else:
                        self.get_logger().warn(f"Multiple objects found ({found_object}) in frame {self.rgbd_frames_counter_processed}, cannot chose")
                        self.initial_detection_counter = 0
                else:
                    # set or reset to 0 if not found
                    self.initial_detection_counter = 0

            if self.initial_detection_counter >= self.min_initial_detection_counter:
                self.initial_detection_counter = 0
                if self.no_fp:
                    self.get_logger().debug(
                        f"Frame {self.rgbd_frames_counter_processed}: skipping FoundationPose (--no_fp)"
                    )
                else:
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


        elif (not self.no_fp) and (self.current_phase == "PoseTracking" or self.current_phase == "StartPoseTracking"):
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
    parser.add_argument("--depth_source", type=str, default="realsense", choices=["realsense", "moge"], help="Depth source: RealSense topic or MoGe monocular depth.")
    parser.add_argument("--moge_checkpoint", type=str, default="./moge2_vitb_normal.pt", help="Path to MoGe-2 checkpoint (.pt).")
    parser.add_argument("--moge_resolution_level", type=int, default=9, help="MoGe resolution level [0-9]; higher is slower/finer.")
    parser.add_argument("--moge_no_camera_fov", action="store_true", default=False, help="Let MoGe estimate FOV instead of using camera_info K (default: pass FOV from K).")
    parser.add_argument("--publish_moge_depth", action="store_true", default=False, help="Publish MoGe depth as sensor_msgs/Image 16UC1 (mm), matching RealSense image_raw.")
    parser.add_argument("--moge_depth_topic", type=str, default="/foundation_pose/moge_depth/image_raw", help="Topic for published MoGe depth image_raw.")
    parser.add_argument("--camera_name", type=str, default="realsense_head_front", help="Camera name.")
    parser.add_argument("--slop", type=float, default=1.0, help="Slop.")
    parser.add_argument("--seg_model_type", type=str, default="yoloe", choices=["yoloe", "sam3", "qwen_sam2_detect", "qwen_sam2_point"], help="Segmentation backend: yoloe, sam3 text, qwen_sam2_detect (Qwen bbox + SAM2), or qwen_sam2_point (Qwen point + SAM2).")
    parser.add_argument("--seg_model_name", type=str, default="yoloe-26s-seg.pt", help="Segmentation weights (YOLOE checkpoint, or SAM2 e.g. sam2_s.pt for qwen_sam2_*; ignored for sam3).")
    parser.add_argument("--yoloe_conf", type=float, default=0.15, help="Confidence threshold for the YOLOE detections.")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3.5-0.8B", help="Hugging Face id or path for the Qwen VLM used by qwen_sam2_*.")
    parser.add_argument("--qwen_thinking_budget", type=int, default=256, help="Max thinking tokens for Qwen before forcing the final answer (0 disables the budget stage).")
    parser.add_argument("--multiple_object_method", type=str, default="none", choices=["none", "biggest"], help="When multiple objects are detected: 'none' skips the frame, 'biggest' keeps the mask with the most pixels.")
    parser.add_argument("--resize_factor", type=int, default=1, help="Resize factor to divide the image size by this factor.")
    parser.add_argument("--min_initial_detection_counter", type=int, default=1, help="Minimum initial detection counter.")
    parser.add_argument("--enable_pose_tracking", action="store_true", default=False, help="Enable pose tracking.")
    parser.add_argument("--no_fp", action="store_true", default=False, help="Skip FoundationPose register/track at runtime (still loads models); detection/segmentation only.")
    parser.add_argument("--publish_mask_image", action="store_true", default=False, help="Publish raw RGB Image with colored detection masks for RViz debug.")
    parser.add_argument("--fp_verbosity", type=str, default="warning", help="Verbosity level for FoundationPose. Valid: debug, info, warning, error, critical.")
    parser.add_argument("--ros_verbosity", type=str, default="info", help="ROS logger severity (debug/info/warn/error/fatal). Defaults to RCUTILS_LOGGING_SEVERITY or INFO.")
    parser.add_argument("--use_onnx", action="store_true", default=False, help="Use ONNX predictors instead of the default scorer and refiner.")
    parser.add_argument("--refiner_onnx", type=str, default="", help="Optional path to refiner_net.onnx.")
    parser.add_argument("--scorer_onnx", type=str, default="", help="Optional path to score_net.onnx.")
    parser.add_argument("--no_prefer_tensorrt", action="store_true", default=False, help="Disable TensorRT EP when using ONNX predictors.")
    args = parser.parse_args()
    args.moge_use_camera_fov = not args.moge_no_camera_fov

    # Set parameters based on the object key
    args.mesh_file = OBJECT_KEYS_TO_PARAMETERS[args.object_key]["mesh_file"]
    args.target_object = OBJECT_KEYS_TO_PARAMETERS[args.object_key]["target_object"]
    args.vlm_prompt = OBJECT_KEYS_TO_PARAMETERS[args.object_key]["vlm_prompt"]
    args.symmetry_x_angles = OBJECT_KEYS_TO_PARAMETERS[args.object_key]["symmetry_x_angles"]
    args.symmetry_y_angles = OBJECT_KEYS_TO_PARAMETERS[args.object_key]["symmetry_y_angles"]
    args.symmetry_z_angles = OBJECT_KEYS_TO_PARAMETERS[args.object_key]["symmetry_z_angles"]

    # Set the parameters based on the camera name
    args.color_topic = f"/rgbd/{args.camera_name}/color/image_raw/compressed"
    args.depth_topic = f"/rgbd/{args.camera_name}/aligned_depth_to_color/image_raw/compressedDepth"
    if args.depth_source == "moge":
        # Color optical frame / color camera_info (no RealSense depth stream required)
        args.camera_info_topic = f"/rgbd/{args.camera_name}/color/camera_info"
        args.pose_frame_id = f"{args.camera_name}_color_optical_frame"
    else:
        args.camera_info_topic = f"/rgbd/{args.camera_name}/aligned_depth_to_color/camera_info"
        args.pose_frame_id = f"{args.camera_name}_depth_optical_frame"

    main(args)

