"""
Pre-build TensorRT engines for FoundationPose ONNX refine/score nets.

Engines are written to foundationpose/weights/onnx/trt_cache via ONNX Runtime's
TensorRT EP (same path used by node.py --use_onnx).

Modes:
  (default)           build for milk batch size (37)
  --all_batches       one dynamic profile covering batch 1..252
  --from_rot_grids    one engine per size in ROT_GRID_BATCH_SIZES below
  --batch_size N ...  custom batch size(s)
"""

from __future__ import annotations

import argparse
import logging
import os
import time


# Default register batch for milk (milk_rot_grid.npy).
_MILK_BATCH = 37
_C_IN = 6
_H = 160
_W = 160
_MAX_BATCH = 252

# Unique len(rot_grid) values from *_rot_grid.npy in the repo root.
# Refresh with:
#   python3 -c "import glob,numpy as np; print(sorted({int(np.load(f).shape[0]) for f in glob.glob('*_rot_grid.npy')}))"
ROT_GRID_BATCH_SIZES = [22, 36, 49, 104, 126, 252]


def _resolve_batch_sizes(args) -> tuple[list[int], bool]:
    """Return (batch_sizes, single_engine_profile)."""
    if args.all_batches:
        return [1, _MAX_BATCH], True
    if args.from_rot_grids:
        sizes = list(ROT_GRID_BATCH_SIZES)
        if args.include_batch_1 and 1 not in sizes:
            sizes = [1] + sizes
        return sizes, False
    if args.batch_size:
        sizes = sorted({int(b) for b in args.batch_size})
        for b in sizes:
            if b < 1 or b > _MAX_BATCH:
                raise ValueError(f"--batch_size values must be in [1, {_MAX_BATCH}], got {b}")
        return sizes, False
    # default: milk
    sizes = [_MILK_BATCH]
    if args.include_batch_1:
        sizes = [1, _MILK_BATCH]
    return sizes, False


def _dummy_inputs(batch: int):
    import torch

    a = torch.randn(batch, _C_IN, _H, _W, device="cuda", dtype=torch.float32)
    b = torch.randn(batch, _C_IN, _H, _W, device="cuda", dtype=torch.float32)
    return a, b


def _model_specs():
    from foundationpose.onnx_predictors import DEFAULT_REFINER_ONNX, DEFAULT_SCORER_ONNX

    return {
        "refiner": {
            "onnx": DEFAULT_REFINER_ONNX,
            "inputs": ["inputA", "inputB"],
            "outputs": ["trans", "rot"],
        },
        "scorer": {
            "onnx": DEFAULT_SCORER_ONNX,
            "inputs": ["inputA", "inputB"],
            "outputs": ["score_logit"],
        },
    }


def _build_model(name: str, cfg: dict, batch_sizes: list[int], single_profile: bool) -> None:
    import torch
    from foundationpose.onnx_predictors import OnnxSession

    onnx_path = cfg["onnx"]
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"Missing ONNX model for {name}: {onnx_path}")

    max_batch = max(batch_sizes)
    logging.info(
        f"=== {name}: {os.path.basename(onnx_path)} | "
        f"mode={'profile 1..'+str(max_batch) if single_profile else 'per-batch'} | "
        f"batches={batch_sizes}"
    )

    session = OnnxSession(
        onnx_path,
        input_names=cfg["inputs"],
        output_names=cfg["outputs"],
        prefer_tensorrt=True,
        max_batch=max_batch,
        single_engine_for_all_batches=single_profile,
    )
    if "TensorrtExecutionProvider" not in session.providers:
        raise RuntimeError(
            f"{name}: TensorRT EP not active after session create: {session.providers}"
        )

    runs = [max_batch] if single_profile else batch_sizes
    if single_profile and 1 not in runs:
        runs = [1] + runs

    for batch in runs:
        logging.info(f"{name}: building/running batch={batch} ...")
        torch.cuda.empty_cache()
        a, b = _dummy_inputs(batch)
        t0 = time.time()
        outs = session.run(a, b)
        elapsed = time.time() - t0
        shapes = [tuple(o.shape) for o in outs]
        logging.info(f"{name}: batch={batch} done in {elapsed:.1f}s, outs={shapes}")

    del session
    torch.cuda.empty_cache()


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    import torch
    from foundationpose.onnx_predictors import DEFAULT_ONNX_DIR

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build TensorRT engines.")

    batch_sizes, single_profile = _resolve_batch_sizes(args)
    cache_dir = os.path.join(DEFAULT_ONNX_DIR, "trt_cache")
    os.makedirs(cache_dir, exist_ok=True)
    logging.info(f"TRT cache dir: {cache_dir}")
    logging.info(f"batch_sizes={batch_sizes} single_profile={single_profile}")

    specs = _model_specs()
    models = []
    if not args.scorer_only:
        models.append(("refiner", specs["refiner"]))
    if not args.refiner_only:
        models.append(("scorer", specs["scorer"]))
    if not models:
        raise ValueError("Nothing to build: both --refiner_only and --scorer_only set.")

    t_all = time.time()
    for name, cfg in models:
        _build_model(name, cfg, batch_sizes, single_profile)
    logging.info(f"All engine builds finished in {time.time() - t_all:.1f}s")
    logging.info(f"Cache contents: {sorted(os.listdir(cache_dir))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-build TensorRT engines for FoundationPose ONNX nets."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--all_batches",
        action="store_true",
        default=False,
        help="Build one dynamic TensorRT profile covering batch 1..252.",
    )
    mode.add_argument(
        "--from_rot_grids",
        action="store_true",
        default=False,
        help=f"Build one engine per size in ROT_GRID_BATCH_SIZES={ROT_GRID_BATCH_SIZES}.",
    )
    mode.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=None,
        help="Custom batch size(s), e.g. --batch_size 1 37 252.",
    )
    parser.add_argument(
        "--include_batch_1",
        action="store_true",
        default=False,
        help="Also build batch=1 (tracking). Useful with default milk / --from_rot_grids.",
    )
    parser.add_argument(
        "--refiner_only",
        action="store_true",
        default=False,
        help="Only build engines for refiner_net.onnx.",
    )
    parser.add_argument(
        "--scorer_only",
        action="store_true",
        default=False,
        help="Only build engines for score_net.onnx.",
    )
    args = parser.parse_args()
    main(args)
