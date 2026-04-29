#!/usr/bin/env python3
"""
Dummy ROS2 subscriber node to debug topic/network/DDS issues.

Modes:
  - rgb:   subscribe to a single RGB CompressedImage topic
  - depth: subscribe to a single depth CompressedImage topic (compressedDepth)
  - sync:  ApproximateTimeSynchronizer(rgb, depth) and print on callback

This mirrors the subscription types used by `node.py` (CompressedImage + message_filters).
"""

import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage
from message_filters import Subscriber, ApproximateTimeSynchronizer


def _stamp_to_float(stamp) -> float:
    # builtin_interfaces/Time: sec + nanosec
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


class DummyNodeSub(Node):
    def __init__(self, *, mode: str, rgb_topic: str, depth_topic: str, slop: float, queue_size: int):
        super().__init__("dummy_node_sub")
        self._mode = mode
        self._rx = 0

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=15,
        )

        if mode == "rgb":
            self._sub = self.create_subscription(
                CompressedImage,
                rgb_topic,
                self._rgb_cb,
                qos_sensor,
            )
            self.get_logger().info(f"[rgb] Subscribed to {rgb_topic} (CompressedImage, BEST_EFFORT)")

        elif mode == "depth":
            self._sub = self.create_subscription(
                CompressedImage,
                depth_topic,
                self._depth_cb,
                qos_sensor,
            )
            self.get_logger().info(f"[depth] Subscribed to {depth_topic} (CompressedImage, BEST_EFFORT)")

        elif mode == "sync":
            sub_rgb = Subscriber(self, CompressedImage, rgb_topic, qos_profile=qos_sensor)
            sub_depth = Subscriber(self, CompressedImage, depth_topic, qos_profile=qos_sensor)
            self._sync = ApproximateTimeSynchronizer(
                [sub_rgb, sub_depth],
                queue_size=queue_size,
                slop=slop,
            )
            self._sync.registerCallback(self._sync_cb)
            self.get_logger().info(
                f"[sync] Subscribed to {rgb_topic} + {depth_topic} (ApproximateTimeSynchronizer queue={queue_size} slop={slop})"
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _rgb_cb(self, msg: CompressedImage):
        self._rx += 1
        t = _stamp_to_float(msg.header.stamp)
        print(
            f"[rgb #{self._rx}] stamp={t:.9f} frame_id='{msg.header.frame_id}' format='{msg.format}' bytes={len(msg.data)}"
        )

    def _depth_cb(self, msg: CompressedImage):
        self._rx += 1
        t = _stamp_to_float(msg.header.stamp)
        print(
            f"[depth #{self._rx}] stamp={t:.9f} frame_id='{msg.header.frame_id}' format='{msg.format}' bytes={len(msg.data)}"
        )

    def _sync_cb(self, rgb_msg: CompressedImage, depth_msg: CompressedImage):
        self._rx += 1
        t_rgb = _stamp_to_float(rgb_msg.header.stamp)
        t_d = _stamp_to_float(depth_msg.header.stamp)
        dt_ms = (t_rgb - t_d) * 1000.0
        print(
            f"[sync #{self._rx}] "
            f"rgb_stamp={t_rgb:.9f} depth_stamp={t_d:.9f} dt_ms={dt_ms:+.3f} | "
            f"rgb_bytes={len(rgb_msg.data)} depth_bytes={len(depth_msg.data)} | "
            f"rgb_format='{rgb_msg.format}' depth_format='{depth_msg.format}'"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rgb", "depth", "sync"], default="sync")
    parser.add_argument("--rgb_topic", type=str, default="/realsense_head_front/color/image_raw/compressed")
    parser.add_argument("--depth_topic", type=str, default="/realsense_head_front/depth/image_rect_raw/compressedDepth")
    parser.add_argument("--slop", type=float, default=1.0)
    parser.add_argument("--queue_size", type=int, default=100)
    args = parser.parse_args()

    rclpy.init()
    node = DummyNodeSub(
        mode=args.mode,
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        slop=args.slop,
        queue_size=args.queue_size,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

