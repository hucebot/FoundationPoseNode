### Realsense camera scripts
These scripts and config are just intended to launch a realsense driver directly from the docker.

```
sh launch_realsense.sh [camera_name]
```

If `[camera_name]` is not specified it will default to `realsense_default`

## Topics note that with the remapping
- Depth topic `/rgbd/camera_name/aligned_depth_to_color/image_raw/compressedDepth`
- Color topic `/rgbd/camera_name/color/image_raw/compressed`