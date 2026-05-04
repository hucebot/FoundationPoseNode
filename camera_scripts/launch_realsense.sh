#!/bin/bash
# Launch RealSense camera node + static TF to remap pointcloud frame

CAMERA_NAME="realsense_test"
NAMESPACE="rgbd"
TARGET_FRAME="head_front_camera_link" # in the tiago TF tree
SOURCE_FRAME="${CAMERA_NAME}_link" # in the realsense TF tree

echo "[realsense] Starting camera node..."
ros2 run realsense2_camera realsense2_camera_node \
  -r __node:=${CAMERA_NAME} -r __ns:=/${NAMESPACE} \
  --ros-args \
  -p enable_color:=true \
  -p enable_depth:=true \
  -p depth_module.depth_profile:="640x480x15" \
  -p rgb_camera.color_profile:="640x480x15" \
  -p camera_name:="realsense_test" \
  -p pointcloud.enable:=true \
  -p align_depth.enable:=true \
  -p enable_infra1:=false \
  -p enable_infra2:=false \
  -p enable_gyro:=false \
  -p enable_accel:=false \
  -p depth_module.emitter_enabled:=1 \
  -p depth_module.visual_preset:=5 \
  -p publish_tf:=true \
  -p unite_imu_method:=0 \
  -p rgb_camera.enable_auto_exposure:=true \
  -p pointcloud.ordered_pc:=true \
  -p pointcloud.stream_filter:=1 &

CAMERA_PID=$!
echo "[realsense] Camera node PID: ${CAMERA_PID}"

# Wait for the node to be up before publishing TF
echo "[realsense] Waiting for node to come up..."
until ros2 node list 2>/dev/null | grep -q "/${NAMESPACE}/camera"; do
  sleep 0.5
done
echo "[realsense] Node up."

echo "[realsense] Publishing static TF: ${SOURCE_FRAME} -> ${TARGET_FRAME}"
ros2 run tf2_ros static_transform_publisher \
  --frame-id "${TARGET_FRAME}" \
  --child-frame-id "${SOURCE_FRAME}" &

TF_PID=$!
echo "[realsense] Static TF PID: ${TF_PID}"

# Trap Ctrl+C to cleanly kill both processes
trap "echo '[realsense] Shutting down...'; kill ${CAMERA_PID} ${TF_PID} 2>/dev/null; exit 0" SIGINT SIGTERM

wait ${CAMERA_PID}
