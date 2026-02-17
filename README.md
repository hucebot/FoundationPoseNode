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
- `seg_model_name`
- `resize_factor`
- `min_initial_detection_counter`
- `enable_pose_tracking`