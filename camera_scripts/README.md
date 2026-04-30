### Realsense camera scripts
These scripts and config are just intended to launch a realsense driver directly from the docker.

```
sh launch_realsense.sh ./camera_config.yaml
```

## Topics note that with the remapping
- Depth topic `/realsense_head_front/camera/aligned_depth_to_color/image_raw/compressedDepth`
- Color topic `/realsense_head_front/camera/color/image_raw/compressed`