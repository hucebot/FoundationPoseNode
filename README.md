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

## Run
```
bash ./docker/run_container.sh 
python node.py
python node.py --resize_factor 2 --mesh_file ./assets/milk/ref_mesh.obj -sya 30,60,90,120,150,180,210,240,270,300,330
python node.py --resize_factor 2 --mesh_file ./assets/milk/ref_mesh.obj -sya 30,60,90,120,150,180,210,240,270,300,330 -frc Force0 --seg_model_type sam3 --target_object white\ bottle
```

To switch objects : 
```
ros2 topic pub /orchestrator/pose/target_object std_msgs/msg/String data:\ \'mesh_update_mustard\' --once
```

To switch the target object (but keep the same mesh)
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
- `seg_model_type` : either `yolo` (will use a YOLO-Seg model with COCO-classes, then the `target_object` must be one of those classes, e.g. `bottle`) or `sam3` (open vocabulary)
- `seg_model_name` : will always default to `sam3.pt` if seg_model_type is sam3 otherwise use to specify the YOLO-Seg model size e.g. `yolo26n-seg.pt`
- `resize_factor` : divide the image size by this factor to reduce memory usage
- `min_initial_detection_counter` : requires minimum consecutive detections (with one and only one valid object in the frame) at the begining before starting the pose estimation (only usable when `seg_model_type` is `yolo`)
- `enable_pose_tracking` : do pose tracking, otherwise keep re-doing the initial pose estimation for each frame
- `fix_rotation_convention` : change the object yaw rotation (around its `z` axis) with 4 options :
  - `None` : Keep the model output
  - `Initial` : when the node is turned on will set an offset to have the yaw in 0-90 deg and keep applying this offset
  - `All` : at every iteration, will offset the yaw to be in 0-90 deg
  - `Force0` : at every iteration will put the yaw to 0 (e.g. for round objects)
- `symmetry_yaw_angles`: intialize symmetry transforms on the `z` axis for Foundation Pose (reduces the number of initial hypothesis), example : `0,90,180,270` for a "square" based object