# Foundation Pose Node

> **MoGe depth (`--depth_source moge`)** — download [Ruicheng/moge-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal) as `moge2_vitb_normal.pt` at the repo root, and place the [microsoft/MoGe](https://github.com/microsoft/MoGe) source under `./moge`. See [Models](#moge-2-required-for---depth_source-moge) for details.

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

Optional monocular depth from [MoGe](https://github.com/microsoft/MoGe) (MoGe-2):
```bibtex
@misc{wang2025moge2,
      title={MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details},
      author={Ruicheng Wang and Sicheng Xu and Yue Dong and Yu Deng and Jianfeng Xiang and Zelong Lv and Guangzhong Sun and Xin Tong and Jiaolong Yang},
      year={2025},
      eprint={2507.02546},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.02546},
}
```

## Summary of changes

1. Detection and segmentation
This repository integrates FoundationPose with a detection and segmentation model (from `ultralytics` for now)

2. ROS Node
The pipeline is turned into a ROS2 Node with the following input and output :

- Input : RGB (`CompressedImage`), Depth (`CompressedImage`, when `--depth_source realsense`), camera intrinsics (`CameraInfo`), object mesh model (`.obj` file)
- Output : 6D Pose (`PoseStamped`); optional mask overlay and MoGe depth `image_raw`

3. Optional MoGe depth
With `--depth_source moge`, RealSense depth is replaced by [MoGe-2](https://github.com/microsoft/MoGe) metric depth from RGB only (color topic + camera_info).

4. Simplified Dockerfile
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

### MoGe-2 (required for `--depth_source moge`)

Download the [Ruicheng/moge-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal) checkpoint and place it at the repo root as:

```
moge2_vitb_normal.pt
```

```
# from Hugging Face (filename in the repo is model.pt)
wget -O moge2_vitb_normal.pt \
  https://huggingface.co/Ruicheng/moge-2-vitb-normal/resolve/main/model.pt
```

Also put the MoGe source tree under `./moge` (e.g. clone [microsoft/MoGe](https://github.com/microsoft/MoGe) into that folder). The node adds `./moge` to `sys.path` and loads `moge.model.v2.MoGeModel`. Docker images install the `utils3d` dependency used by MoGe.

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

### MoGe depth instead of RealSense
Requires `moge2_vitb_normal.pt` and `./moge` (see Models above). Subscribes to color + camera_info only; publishes poses in the color optical frame.
```
python node.py --object_key milk --depth_source moge --publish_moge_depth --ros_verbosity info
```
Optional: `--moge_checkpoint ./moge2_vitb_normal.pt`, `--moge_resolution_level 5` (lower is faster), `--moge_no_camera_fov`, `--moge_depth_topic /foundation_pose/moge_depth/image_raw`.

With `--publish_moge_depth`, MoGe depth is published as `sensor_msgs/Image` with `encoding: 16UC1` (millimeters), matching RealSense `image_raw` for easy RViz display.

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
ros2 topic pub /orchestrator/foundation_pose/toggle_tracking std_msgs/msg/Bool data:\ true --once
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

## Parameters

- `object_key` : object preset from `OBJECT_KEYS_TO_PARAMETERS` (sets mesh, text prompt, symmetries, and rotation constraints)
- `camera_name` : builds color/depth/camera_info topics and `pose_frame_id` under `/rgbd/<camera_name>/...`
- `est_refine_iter` / `track_refine_iter` : FoundationPose refine iters for register / track
- `debug` / `debug_dir` : debug level and output directory (auto path if empty)
- `depth_scale` : depth units to meters (default `0.001`)
- `depth_source` : `realsense` (synced RGB+depth) or `moge` (RGB-only + MoGe-2 metric depth)
- `moge_checkpoint` : path to MoGe-2 weights (default `./moge2_vitb_normal.pt`)
- `moge_resolution_level` : MoGe detail level `[0-9]` (default `9`; lower is faster)
- `moge_no_camera_fov` : let MoGe estimate FOV instead of using `camera_info` K
- `publish_moge_depth` : publish MoGe depth as `16UC1` mm `image_raw` (requires `--depth_source moge`)
- `moge_depth_topic` : MoGe depth topic (default `/foundation_pose/moge_depth/image_raw`)
- `slop` : RGB–depth sync slop (RealSense path only)
- `seg_model_type` : `yoloe` or `sam3` (open-vocab; prompt comes from the object preset)
- `seg_model_name` : YOLOE weights e.g. `yoloe-26s-seg.pt` (ignored for sam3)
- `yoloe_conf` : YOLOE confidence threshold (default `0.15`)
- `resize_factor` : divide image size by this to save memory
- `min_initial_detection_counter` : consecutive single-object detections required before starting pose (yoloe)
- `enable_pose_tracking` : track after first pose; otherwise re-register every frame
- `publish_mask_image` : publish RGB with colored masks for RViz
- `fp_verbosity` / `ros_verbosity` : FoundationPose / ROS log levels
- `use_onnx` : use ONNX refine/score instead of default predictors
- `refiner_onnx` / `scorer_onnx` : optional paths to ONNX models
- `no_prefer_tensorrt` : disable TensorRT EP when using ONNX

Per-object fields in `OBJECT_KEYS_TO_PARAMETERS` (not CLI): `symmetry_x/y/z_angles` (comma-separated deg, fewer hypotheses), `constraint_yaw/pitch/roll_in` (`None`, `0`, or `[lo,hi]` to clamp Euler angles after pose).

> [!WARNING]
> When using --use_onnx by default the node will try to instantiate TensorRT engines. They will be saved in `foundationpose/weights/onnx/trt_cache`.
> In `onnx_predictors.py` the variable `_TRT_SINGLE_ENGINE_FOR_ALL_BATCHES = False` controls how these engines are created. If `False` engine creation should take 
> a from one to a few minutes (depending on the `batch_size` which is related to the number of hypothesis, the more symmetries the smaller the `batch_size`), if `True` it could be much longer (creating a single engine for all batch sizes) but then you should be able to change objects including 
> objects with different symmetries without a new engine being created.
> If you want to use ONNX-Runtime without TensorRT you can pass the flag `--no_prefer_tensorrt`.
> Also note that creating the engine can take a lot VRAM.


# Realsense camera driver
To launch a realsense camera from the docker :

```
./launch_realsense.sh [camera_name]
```

If `[camera_name]` is not specified it will default to `realsense_default`
