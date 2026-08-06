"""
Run a blank pass of the Ultralytics detectors used by node.py (YOLOE and SAM3).

The first use of these models triggers lazy work that would otherwise happen on the first
run of the node: pip installs done by ultralytics itself (CLIP, mobileclip, ...) and the
download of the checkpoints and of the text encoder. Running this once at image build time
gets it out of the way, so the node starts offline and without surprises.

Downloads are written to `weights_dir`, which is also registered as the ultralytics
`weights_dir` setting so that the assets are found later whatever the working directory is.
"""

import argparse
import os
import sys

import numpy as np


def main(args):
    os.makedirs(args.weights_dir, exist_ok=True)

    from ultralytics.utils import SETTINGS

    SETTINGS.update({"weights_dir": args.weights_dir})
    # ultralytics downloads assets relative to the working directory when they are missing
    os.chdir(args.weights_dir)
    print(f"[initialize_detectors] weights directory: {args.weights_dir}")

    blank_image = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
    failures = []

    if not args.skip_yoloe:
        from ultralytics import YOLOE

        for model_name in args.yoloe_models:
            print(f"[initialize_detectors] YOLOE: {model_name}")
            try:
                model = YOLOE(model_name)
                # set_classes() pulls the text encoder (mobileclip) and its dependencies
                model.set_classes([args.target_object])
                model.predict(blank_image, device=args.device, verbose=False)
                print(f"[initialize_detectors] YOLOE {model_name} OK")
            except Exception as e:
                failures.append(f"YOLOE {model_name}: {e}")
                print(f"[initialize_detectors] YOLOE {model_name} FAILED: {e}")

    if not args.skip_sam3:
        print(f"[initialize_detectors] SAM3: {args.sam3_model}")
        try:
            # the import alone installs the SAM3 dependencies of ultralytics
            from ultralytics.models.sam import SAM3SemanticPredictor

            overrides = dict(
                conf=0.25,
                imgsz=644,
                task="segment",
                mode="predict",
                model=args.sam3_model,
                device=args.device,
                half=args.device != "cpu",
                save=False,
                verbose=False,
            )
            predictor = SAM3SemanticPredictor(overrides=overrides)
            predictor.set_image(blank_image)
            predictor(text=[args.target_object], verbose=False)
            print(f"[initialize_detectors] SAM3 {args.sam3_model} OK")
        except Exception as e:
            # SAM3 weights are gated on Hugging Face and cannot be fetched during a build
            failures.append(f"SAM3 {args.sam3_model}: {e}")
            print(f"[initialize_detectors] SAM3 {args.sam3_model} FAILED: {e}")

    print(f"[initialize_detectors] content of {args.weights_dir}: {sorted(os.listdir(args.weights_dir))}")
    if failures:
        print(f"[initialize_detectors] {len(failures)} detector(s) not initialized:")
        for failure in failures:
            print(f"  - {failure}")
        if args.strict:
            sys.exit(1)
    else:
        print("[initialize_detectors] all detectors initialized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blank pass of the ultralytics detectors to trigger their installs and downloads.")
    parser.add_argument("--weights_dir", type=str, default="/opt/ultralytics_weights", help="Directory where the ultralytics assets are downloaded (also set as the ultralytics weights_dir setting).")
    parser.add_argument("--yoloe_models", type=str, nargs="+", default=["yoloe-26s-seg.pt"], help="YOLOE checkpoints to initialize.")
    parser.add_argument("--sam3_model", type=str, default="sam3/sam3.pt", help="SAM3 checkpoint to initialize (gated weights, skipped with a warning if unavailable).")
    parser.add_argument("--target_object", type=str, default="object", help="Text prompt used for the blank pass.")
    parser.add_argument("--image_size", type=int, default=640, help="Side of the square blank image fed to the models.")
    parser.add_argument("--device", type=str, default="cpu", help="Device used for the blank pass (cpu, since no GPU is available during a docker build).")
    parser.add_argument("--skip_yoloe", action="store_true", help="Do not initialize YOLOE.")
    parser.add_argument("--skip_sam3", action="store_true", help="Do not initialize SAM3.")
    parser.add_argument("--strict", action="store_true", help="Exit with an error if a detector could not be initialized.")
    args = parser.parse_args()
    main(args)
