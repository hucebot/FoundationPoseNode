#!/bin/bash
# Launch RealSense camera node + static TF to remap pointcloud frame

NAMESPACE="realsense_head_front"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/camera_config.yaml}"
TARGET_FRAME="head_front_camera_link"
SOURCE_FRAME="camera_link"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[realsense] ERROR: config file not found: ${CONFIG}" >&2
  echo "[realsense] Usage: $(basename "$0") [/path/to/camera_config.yaml]" >&2
  exit 1
fi

echo "[realsense] Starting camera node..."
ros2 run realsense2_camera realsense2_camera_node \
  --ros-args \
  --remap __ns:=/${NAMESPACE} \
  --params-file "${CONFIG}" &

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
