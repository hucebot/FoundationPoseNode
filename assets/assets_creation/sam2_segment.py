#!/usr/bin/env python3
"""Interactive SAM2 point-prompt segmentation (OpenCV UI).

Usage:
    python sam2_segment.py path/to/image.jpg

Controls:
    Left click   positive point
    Right click  negative point
    s            save mask and start next object
    r            reset current points
    q / ESC      quit
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM


class Sam2Segmenter:
    def __init__(self, image_path: str, model_name: str = "sam2.1_b.pt"):
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        self.points: list[list[int]] = []
        self.labels: list[int] = []
        self.mask: np.ndarray | None = None
        self.object_idx = 0

        self.output_dir = self.image_path.parent / self.image_path.stem
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading {model_name}...")
        self.model = SAM(model_name)
        self.window = f"SAM2 - {self.image_path.name}"

    def predict(self):
        if not self.points:
            self.mask = None
            return

        results = self.model(
            str(self.image_path),
            points=[self.points],
            labels=[self.labels],
            verbose=False,
        )
        masks = results[0].masks
        if masks is None:
            self.mask = None
            return
        self.mask = masks.data[0].cpu().numpy() > 0.5

    def render(self) -> np.ndarray:
        vis = self.image.copy()
        if self.mask is not None:
            vis[self.mask] = (vis[self.mask] * 0.5 + np.array([0, 255, 0])).astype(np.uint8)
        for (x, y), label in zip(self.points, self.labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(vis, (x, y), 5, color, -1)
            cv2.circle(vis, (x, y), 6, (255, 255, 255), 1)
        return vis

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            self.labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append([x, y])
            self.labels.append(0)
        else:
            return
        self.predict()

    def save_mask(self):
        if self.mask is None:
            print("No mask to save — add at least one positive point.")
            return
        out_path = self.output_dir / f"{self.object_idx}.png"
        cv2.imwrite(str(out_path), (self.mask.astype(np.uint8) * 255))
        print(f"Saved {out_path}")
        self.object_idx += 1
        self.points.clear()
        self.labels.clear()
        self.mask = None

    def reset(self):
        self.points.clear()
        self.labels.clear()
        self.mask = None

    def run(self):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.on_mouse)
        print(__doc__)
        print(f"Output dir: {self.output_dir}")

        while True:
            cv2.imshow(self.window, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                self.save_mask()
            if key == ord("r"):
                self.reset()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive SAM2 segmentation")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--model", default="sam2.1_b.pt", help="SAM2 weights (default: sam2.1_b.pt)")
    args = parser.parse_args()
    Sam2Segmenter(args.image_path, args.model).run()


if __name__ == "__main__":
    main()
