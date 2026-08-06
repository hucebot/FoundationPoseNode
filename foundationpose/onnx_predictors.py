"""ONNX Runtime predictors for NVIDIA TAO FoundationPose deployable_v1.0 nets.

Models (NGC: nvidia/tao/foundationpose:deployable_v1.0):
  - refiner_net.onnx  inputs: inputA, inputB  [B,6,160,160]  outputs: trans, rot [B,3]
  - score_net.onnx    inputs: inputA, inputB  [B,6,160,160]  outputs: score_logit [B,1]

Same crop / render preprocessing as the PyTorch predictors; only the network
forward is replaced by ONNX Runtime (CUDA / TensorRT EP when available).
"""

from __future__ import annotations

import logging
import os
import sys
import time
import ctypes.util
import ctypes
import glob
from typing import Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from learning.training.predict_pose_refine import make_crop_data_batch as make_refine_crop_batch
from learning.training.predict_score import make_crop_data_batch as make_score_crop_batch
from learning.datasets.h5_dataset import PoseRefinePairH5Dataset, ScoreMultiPairH5Dataset
from Utils import *


DEFAULT_ONNX_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "onnx"
)
DEFAULT_REFINER_ONNX = os.path.join(DEFAULT_ONNX_DIR, "refiner_net.onnx")
DEFAULT_SCORER_ONNX = os.path.join(DEFAULT_ONNX_DIR, "score_net.onnx")

# Matches foundationpose/weights/*/config.yml for the deployable TAO nets
_REFINE_CFG = {
    "c_in": 6,
    "zfar": float("inf"),
    "use_BN": True,
    "rot_rep": "axis_angle",
    "trans_rep": "tracknet",
    "crop_ratio": 1.2,
    "use_normal": False,
    "use_mask": False,
    "input_resize": [160, 160],
    "normalize_xyz": True,
    "rot_normalizer": 0.3490658503988659,
    "trans_normalizer": [0.02, 0.02, 0.05],
    "n_view": 1,
}
_SCORE_CFG = {
    "c_in": 6,
    "zfar": float("inf"),
    "use_BN": True,
    "crop_ratio": 1.1,
    "use_normal": False,
    "use_mask": False,
    "input_resize": [160, 160],
    "normalize_xyz": True,
    "n_view": 1,
}


def _default_ort_providers(prefer_tensorrt: bool = True) -> list:
    """Pick ORT execution providers: TensorRT > CUDA > CPU."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    providers = []

    # ORT may report the TensorRT EP as available because the wheel was built
    # with it, even when the container does not ship the TensorRT runtime libs.
    # In that case, explicitly requesting TRT makes ORT emit loud load errors and
    # may fall back less gracefully than going straight to CUDA.
    trt_lib_dir = _find_tensorrt_lib_dir()
    has_trt_runtime = trt_lib_dir is not None
    if prefer_tensorrt and "TensorrtExecutionProvider" in available and has_trt_runtime:
        _preload_tensorrt_runtime(trt_lib_dir)
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": os.path.join(DEFAULT_ONNX_DIR, "trt_cache"),
                },
            )
        )
    elif prefer_tensorrt and "TensorrtExecutionProvider" in available and not has_trt_runtime:
        raise RuntimeError("TensorRT runtime libraries are missing in this container. Use --no_prefer_tensorrt to use CUDAExecutionProvider instead.")
    elif prefer_tensorrt and "TensorrtExecutionProvider" not in available:
        raise RuntimeError("TensorRT execution provider is not available. Use --no_prefer_tensorrt to use CUDAExecutionProvider instead.")
    
    if "CUDAExecutionProvider" in available:
        providers.append(
            (
                "CUDAExecutionProvider",
                {"device_id": 0, "arena_extend_strategy": "kNextPowerOfTwo"},
            )
        )
    providers.append("CPUExecutionProvider")
    return providers


def _find_tensorrt_lib_dir() -> Optional[str]:
    """Locate pip or system TensorRT shared libraries."""
    candidates = []

    found = ctypes.util.find_library("nvinfer")
    if found:
        # The system loader can already find it, so no explicit preload dir is needed.
        return ""

    try:
        import tensorrt_libs  # type: ignore

        candidates.append(os.path.dirname(tensorrt_libs.__file__))
    except Exception:
        pass

    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates.extend(
        [
            f"/usr/local/lib/{pyver}/dist-packages/tensorrt_libs",
            f"/usr/local/lib/{pyver}/site-packages/tensorrt_libs",
            f"/usr/local/lib/{pyver}/dist-packages/nvidia/tensorrt_libs",
            f"/usr/local/lib/{pyver}/site-packages/nvidia/tensorrt_libs",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
        ]
    )
    candidates.extend(glob.glob("/usr/local/lib/python*/dist-packages/tensorrt_libs"))
    candidates.extend(glob.glob("/usr/local/lib/python*/site-packages/tensorrt_libs"))

    for directory in candidates:
        if not directory:
            continue
        if os.path.isfile(os.path.join(directory, "libnvinfer.so.10")):
            return directory
        if glob.glob(os.path.join(directory, "libnvinfer.so*")):
            return directory
    return None


def _preload_tensorrt_runtime(trt_lib_dir: str) -> None:
    """Load TensorRT libs from pip/system location so ORT can enable the TRT EP."""
    if trt_lib_dir:
        os.environ["LD_LIBRARY_PATH"] = f"{trt_lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        for lib_name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
            lib_path = os.path.join(trt_lib_dir, lib_name)
            if os.path.isfile(lib_path):
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)


class OnnxSession:
    """Thin onnxruntime wrapper; torch CUDA tensors in/out."""

    def __init__(
        self,
        onnx_path: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        providers: Optional[list] = None,
        prefer_tensorrt: bool = True,
    ):
        import onnxruntime as ort

        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                "Download TAO deployable_v1.0 with:\n"
                "  mkdir -p foundationpose/weights/onnx && cd foundationpose/weights/onnx && \\\n"
                "  curl -L -o refiner_net.onnx "
                "https://api.ngc.nvidia.com/v2/models/nvidia/tao/foundationpose/versions/deployable_v1.0/files/refiner_net.onnx && \\\n"
                "  curl -L -o score_net.onnx "
                "https://api.ngc.nvidia.com/v2/models/nvidia/tao/foundationpose/versions/deployable_v1.0/files/score_net.onnx"
            )
        self.onnx_path = onnx_path
        self.input_names = list(input_names)
        self.output_names = list(output_names)
        if providers is None:
            providers = _default_ort_providers(prefer_tensorrt=prefer_tensorrt)
        os.makedirs(os.path.join(DEFAULT_ONNX_DIR, "trt_cache"), exist_ok=True)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        t0 = time.time()
        logging.info(f"Creating ONNX Runtime session with providers={providers}")
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.providers = self.session.get_providers()
        logging.info(
            f"Loaded ONNX {os.path.basename(onnx_path)} in {time.time()-t0:.2f}s "
            f"providers={self.providers}"
        )

    def run(self, *tensors: torch.Tensor) -> list[torch.Tensor]:
        """Run with NCHW float32 torch tensors; returns torch CUDA float tensors."""
        feeds = {}
        for name, t in zip(self.input_names, tensors):
            if t.dtype != torch.float32:
                t = t.float()
            feeds[name] = t.detach().contiguous().cpu().numpy()
        outs = self.session.run(self.output_names, feeds)
        return [torch.as_tensor(o, device="cuda", dtype=torch.float32) for o in outs]


class PoseRefinePredictorOnnx:
    """Drop-in replacement for PoseRefinePredictor using TAO refiner_net.onnx."""

    def __init__(
        self,
        onnx_path: str = DEFAULT_REFINER_ONNX,
        prefer_tensorrt: bool = True,
        providers: Optional[list] = None,
        infer_batch_size: int = 252,
    ):
        logging.info("PoseRefinePredictorOnnx init")
        self.cfg = OmegaConf.create(_REFINE_CFG)
        self.dataset = PoseRefinePairH5Dataset(cfg=self.cfg, h5_file="", mode="test")
        self.session = OnnxSession(
            onnx_path,
            input_names=["inputA", "inputB"],
            output_names=["trans", "rot"],
            providers=providers,
            prefer_tensorrt=prefer_tensorrt,
        )
        self.infer_batch_size = int(infer_batch_size)
        self.last_trans_update = None
        self.last_rot_update = None
        # Compatibility with FoundationPose.to_device (no nn.Module)
        self.model = self
        logging.info("PoseRefinePredictorOnnx init done")

    def to(self, *args, **kwargs):
        return self

    def cuda(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    @torch.inference_mode()
    def predict(
        self,
        rgb,
        depth,
        K,
        ob_in_cams,
        xyz_map,
        normal_map=None,
        get_vis=False,
        mesh=None,
        mesh_tensors=None,
        glctx=None,
        mesh_diameter=None,
        iteration=5,
    ):
        logging.info(f"ob_in_cams:{ob_in_cams.shape}")
        tf_to_center = np.eye(4)
        if not self.cfg.use_normal:
            normal_map = None

        crop_ratio = self.cfg["crop_ratio"]
        B_in_cams = torch.as_tensor(ob_in_cams, device="cuda", dtype=torch.float)

        if mesh_tensors is None:
            mesh_tensors = make_mesh_tensors(mesh)

        rgb_tensor = torch.as_tensor(rgb, device="cuda", dtype=torch.float)
        depth_tensor = torch.as_tensor(depth, device="cuda", dtype=torch.float)
        xyz_map_tensor = torch.as_tensor(xyz_map, device="cuda", dtype=torch.float)
        bs = self.infer_batch_size

        for _ in range(iteration):
            pose_data = make_refine_crop_batch(
                self.cfg.input_resize,
                B_in_cams,
                mesh,
                rgb_tensor,
                depth_tensor,
                K,
                crop_ratio=crop_ratio,
                normal_map=normal_map,
                xyz_map=xyz_map_tensor,
                cfg=self.cfg,
                glctx=glctx,
                mesh_tensors=mesh_tensors,
                dataset=self.dataset,
                mesh_diameter=mesh_diameter,
            )
            refined = []
            n = pose_data.rgbAs.shape[0]
            for b in range(0, n, bs):
                A = torch.cat(
                    [pose_data.rgbAs[b : b + bs], pose_data.xyz_mapAs[b : b + bs]], dim=1
                ).float()
                B = torch.cat(
                    [pose_data.rgbBs[b : b + bs], pose_data.xyz_mapBs[b : b + bs]], dim=1
                ).float()
                trans, rot = self.session.run(A, B)
                # TAO / normalize_xyz path (same as config.yml)
                trans_delta = trans * (mesh_diameter / 2)
                rot_mat_delta = torch.tanh(rot) * float(self.cfg["rot_normalizer"])
                rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0, 2, 1)
                refined.append(
                    egocentric_delta_pose_to_pose(
                        pose_data.poseA[b : b + bs],
                        trans_delta=trans_delta,
                        rot_mat_delta=rot_mat_delta,
                    )
                )
            B_in_cams = torch.cat(refined, dim=0).reshape(len(ob_in_cams), 4, 4)

        B_in_cams_out = B_in_cams @ torch.tensor(
            tf_to_center[None], device="cuda", dtype=torch.float
        )
        self.last_trans_update = trans_delta
        self.last_rot_update = rot_mat_delta
        torch.cuda.empty_cache()
        if get_vis:
            return B_in_cams_out, None
        return B_in_cams_out, None


class ScorePredictorOnnx:
    """Drop-in replacement for ScorePredictor using TAO score_net.onnx."""

    def __init__(
        self,
        amp=True,
        onnx_path: str = DEFAULT_SCORER_ONNX,
        prefer_tensorrt: bool = True,
        providers: Optional[list] = None,
        infer_batch_size: int = 252,
    ):
        logging.info("ScorePredictorOnnx init")
        self.amp = amp
        self.cfg = OmegaConf.create(_SCORE_CFG)
        self.dataset = ScoreMultiPairH5Dataset(
            cfg=self.cfg, mode="test", h5_file=None, max_num_key=1
        )
        self.session = OnnxSession(
            onnx_path,
            input_names=["inputA", "inputB"],
            output_names=["score_logit"],
            providers=providers,
            prefer_tensorrt=prefer_tensorrt,
        )
        self.infer_batch_size = int(infer_batch_size)
        self.model = self
        logging.info("ScorePredictorOnnx init done")

    def to(self, *args, **kwargs):
        return self

    def cuda(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    @torch.inference_mode()
    def predict(
        self,
        rgb,
        depth,
        K,
        ob_in_cams,
        normal_map=None,
        get_vis=False,
        mesh=None,
        mesh_tensors=None,
        glctx=None,
        mesh_diameter=None,
    ):
        logging.info(f"ob_in_cams:{ob_in_cams.shape}")
        ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device="cuda")
        if not self.cfg.use_normal:
            normal_map = None
        if mesh_tensors is None:
            mesh_tensors = make_mesh_tensors(mesh)

        rgb = torch.as_tensor(rgb, device="cuda", dtype=torch.float)
        depth = torch.as_tensor(depth, device="cuda", dtype=torch.float)

        pose_data = make_score_crop_batch(
            self.cfg.input_resize,
            ob_in_cams,
            mesh,
            rgb,
            depth,
            K,
            crop_ratio=self.cfg["crop_ratio"],
            glctx=glctx,
            mesh_tensors=mesh_tensors,
            dataset=self.dataset,
            cfg=self.cfg,
            mesh_diameter=mesh_diameter,
        )

        n = pose_data.rgbAs.shape[0]
        scores = torch.zeros((n,), dtype=torch.float, device="cuda")
        bs = self.infer_batch_size
        for b in range(0, n, bs):
            A = torch.cat(
                [pose_data.rgbAs[b : b + bs], pose_data.xyz_mapAs[b : b + bs]], dim=1
            ).float()
            B = torch.cat(
                [pose_data.rgbBs[b : b + bs], pose_data.xyz_mapBs[b : b + bs]], dim=1
            ).float()
            (logit,) = self.session.run(A, B)
            scores[b : b + bs] = logit.float().reshape(-1)

        torch.cuda.empty_cache()
        if get_vis:
            return scores, None
        return scores, None
