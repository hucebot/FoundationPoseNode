# Foundation Pose Node

## FoundationPose
Derived from [FoundationPose](github.com/NVlabs/FoundationPose)

![Illustration](illustrations/6dbottlerviz.gif)

```bibtex
@InProceedings{foundationposewen2024,
author        = {Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield},
title         = {{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
booktitle     = {CVPR},
year          = {2024},
}
```

```bibtex
@InProceedings{bundlesdfwen2023,
author        = {Bowen Wen and Jonathan Tremblay and Valts Blukis and Stephen Tyree and Thomas M\"{u}ller and Alex Evans and Dieter Fox and Jan Kautz and Stan Birchfield},
title         = {{BundleSDF}: {N}eural 6-{DoF} Tracking and {3D} Reconstruction of Unknown Objects},
booktitle     = {CVPR},
year          = {2023},
}
```

Using models of Ultralytics such as
```bibtex
@software{yolo26_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO26},
  version = {26.0.0},
  year = {2026},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```

## Summary of changes

1. Detection and segmentation
This repository integrates FoundationPose with a detection and segmentation model (from `ultralytics` for now)

2. ROS Node
The pipeline is turned into a ROS2 Node with the following input and output :

- Input : RGB (`CompressedImage`), Depth (`CompressedImage`), camera intrinsics (`CameraInfo`), object mesh model (`.obj` file)
- Output : 6D Pose (`PoseStamped`)

3. Simplified Dockerfile
It includes a simplified Dockerfile compared to the original version (does not using conda in the docker). The installation contains FoundationPose dependencies and ROS dependencies.

## Install
```
docker build --network host -f docker/dockerfile -t foundationposev2 .
```

On a Jetson AGX Thor Developer Kit (JetPack 7.2 GA / L4T 39.2.0) use the `docker_jetson` variant instead. It targets Ubuntu 24.04, CUDA 13, ROS2 Jazzy and the `sm_110` Blackwell GPU:
```
docker build --network host -f docker_jetson/dockerfile -t foundationposev2_jetson .
bash ./docker_jetson/run_container.sh
```

## Models

Weights are not shipped in this repo (they are gitignored). Download them before running the node.

### FoundationPose

Download the network weights from the [FoundationPose Google Drive](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) and place them under `foundationpose/weights/`:

```
foundationpose/weights/
├── 2023-10-28-18-33-37/   # pose refiner
│   ├── config.yml
│   └── model_best.pth
└── 2024-01-11-20-02-45/   # pose scorer
    ├── config.yml
    └── model_best.pth
```

### FoundationPose ONNX (TAO deployable_v1.0)

For `node.py --use_onnx` / `benchmark_onnx.py`, download the [NGC TAO FoundationPose](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/foundationpose) ONNX nets:

```
mkdir -p foundationpose/weights/onnx && cd foundationpose/weights/onnx
curl -L -o refiner_net.onnx \
  https://api.ngc.nvidia.com/v2/models/nvidia/tao/foundationpose/versions/deployable_v1.0/files/refiner_net.onnx
curl -L -o score_net.onnx \
  https://api.ngc.nvidia.com/v2/models/nvidia/tao/foundationpose/versions/deployable_v1.0/files/score_net.onnx
```

Requires `onnxruntime-gpu` in the image (added at the end of `docker/dockerfile` — rebuild the image once after pulling that change).

### SAM3 (`--seg_model_type sam3`)

SAM3 weights are gated and are **not** auto-downloaded by Ultralytics. Request access on [Hugging Face: facebook/sam3](https://huggingface.co/facebook/sam3), then download [`sam3.pt`](https://huggingface.co/facebook/sam3/resolve/main/sam3.pt?download=true) and place it at:

```
sam3/sam3.pt
```

(relative to the repo root / working directory when you run `node.py`).

### YOLOE (`--seg_model_type yoloe`)

[YOLOE](https://docs.ultralytics.com/models/yoloe) is open vocabulary too: the `target_object` string is used as a text prompt, so it is not restricted to COCO classes.

Ultralytics will usually download the chosen checkpoint (e.g. `yoloe-26s-seg.pt`, or `yoloe-26n/m/l/x-seg.pt` for other sizes) on first use. You can also put it in the working directory (repo root). The text encoder weights (`mobileclip2_b.ts`) are downloaded on first `set_classes()` call.

### Detector pre-initialization

`initialize_detectors.py` runs a blank pass of YOLOE and SAM3 so the pip installs and downloads that Ultralytics does lazily on first use happen once instead of at node startup. Both dockerfiles run it at the end of the build (which is why the build needs `--network host`), downloading into `/opt/ultralytics_weights` and registering that directory as the Ultralytics `weights_dir`, so the node finds the assets from any working directory. SAM3 weights are gated, so that part only prints a warning during the build.

To redo it manually, e.g. for another YOLOE size:
```
python initialize_detectors.py --yoloe_models yoloe-26s-seg.pt yoloe-26l-seg.pt
```

## Run
```
bash ./docker/run_container.sh 
python node.py --resize_factor 2 --object_key milk --seg_model_type sam3 --ros_verbosity info --publish_mask_image
```

### ONNX node (TAO refine/score)
Same arguments as `node.py`, plus `--use_onnx` so refine/score use the NGC ONNX models via ONNX Runtime:
```
python node.py --resize_factor 2 --object_key milk --seg_model_type sam3 --use_onnx --ros_verbosity info --publish_mask_image
```

### Offline ONNX benchmark (no ROS)
```
python benchmark_onnx.py \
  --rgb_file ./illustrations/objects_hackathon2.jpeg \
  --mesh_file ./assets/hackathon2/milk/milk.obj \
  --target_object bottle \
  --iters 5 --warmup 1
```
Without `--depth_file`, a planar fake depth is filled in the detection mask (useful for timing; poses are not metrically accurate).

Toggle the node (on by default)
```
ros2 topic pub /orchestrator/foundation_pose/toggle_fp std_msgs/msg/Bool data:\ true --once
```

To switch objects : 
```
ros2 topic pub /orchestrator/foundation_pose/target_object std_msgs/msg/String data:\ \'mesh_update_mustard\' --once
```

Toggle the tracking on or off (off by default)
```
ros2 topic pub /orchestrator/pose/toggle_tracking std_msgs/msg/Bool data:\ true --once
```

Object names (update the switch objects string according to your need) : 
```
baguette
banana
coffeecan
egg
flowercup
jam
milk
minicheese
pan
redapple
smallmilk
smallsanpellegrino
spam
ycbmustard
```

## Troubleshooting
Quick DDS/topic test (prints from callback only):
```
python dummy_node_sub.py --mode rgb
python dummy_node_sub.py --mode depth
python dummy_node_sub.py --mode sync --slop 0.05
```

To switch the target object string (but keep the same mesh)
```
ros2 topic pub /orchestrator/pose/target_object std_msgs/msg/String data:\ \'yellow\ bottle\' --once
```


## Parameters

- `mesh_file`
- `target_object`
- `est_refine_iter`
- `track_refine_iter`
- `debug`
- `debug_dir`
- `depth_scale`
- `color_topic`
- `depth_topic`
- `camera_info_topic`
- `pose_frame_id`
- `slop`
- `seg_model_type` : either `yoloe` or `sam3`, both are open vocabulary so the `target_object` can be any text prompt (e.g. `yellow bottle`)
- `seg_model_name` : will always default to `sam3.pt` if seg_model_type is sam3 otherwise use to specify the YOLOE model size e.g. `yoloe-26s-seg.pt`
- `yoloe_conf` : confidence threshold of the YOLOE detections, defaults to `0.15` (lower than the ultralytics default of `0.25` since open vocabulary prompts often score low)
- `resize_factor` : divide the image size by this factor to reduce memory usage
- `min_initial_detection_counter` : requires minimum consecutive detections (with one and only one valid object in the frame) at the begining before starting the pose estimation (only usable when `seg_model_type` is `yoloe`)
- `enable_pose_tracking` : do pose tracking, otherwise keep re-doing the initial pose estimation for each frame
- `fix_rotation_convention` : change the object yaw rotation (around its `z` axis) with 4 options :
  - `None` : Keep the model output
  - `Initial` : when the node is turned on will set an offset to have the yaw in 0-90 deg and keep applying this offset
  - `All` : at every iteration, will offset the yaw to be in 0-90 deg
  - `Force0` : at every iteration will put the yaw to 0 (e.g. for round objects)
- `symmetry_yaw_angles`: intialize symmetry transforms on the `z` axis for Foundation Pose (reduces the number of initial hypothesis), example : `0,90,180,270` for a "square" based object


# Realsense camera driver
To launch a realsense camera from the docker :

```
./launch_realsense.sh [camera_name]
```

If `[camera_name]` is not specified it will default to `realsense_default`

## TODO
- [] Handle all kind of symmetries (not only Z axis)
- [] Simplify the cluster_poses for infinite symmetry of round objects
- [] Add another kind of filtering to avoid flipping
- [] Check point-cloud stability for depth estimation (inc. in fridge)
- [] Better referencing of objects for detection