"""
Local Qwen VLM memory probe (same load path as node.py --seg_model_type qwen_sam2).

Loads Qwen, runs one bbox prediction on a demo image, and prints CUDA memory at each
stage so you can see whether Qwen alone (or Qwen+SAM2) is what OOMs the node.

Example:
  python qwen_memory_demo.py
  python qwen_memory_demo.py --image ./illustrations/objects_hackathon2.jpeg --vlm_prompt "Locate the milk"
  python qwen_memory_demo.py --with_sam2
"""

import argparse
import json
import os
import re
import time

import numpy as np
import torch
from PIL import Image as PILImage


def _parse_qwen_bboxes_norm(text: str):
    """Parse Qwen grounding JSON; return list of [x1,y1,x2,y2] in 0-1000 coords."""
    if not text:
        return []
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    candidates = []
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "bbox_2d" in item:
                    bbox = item["bbox_2d"]
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        candidates.append([float(v) for v in bbox])
    except Exception:
        pass

    if not candidates:
        for match in re.finditer(
            r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]',
            text,
        ):
            candidates.append([float(match.group(i)) for i in range(1, 5)])
    return candidates


def _norm1000_bbox_to_xyxy(bbox_norm, width: int, height: int):
    """Convert a Qwen 0-1000 bbox to clipped integer pixel xyxy."""
    x1 = int(round(bbox_norm[0] / 1000.0 * width))
    y1 = int(round(bbox_norm[1] / 1000.0 * height))
    x2 = int(round(bbox_norm[2] / 1000.0 * width))
    y2 = int(round(bbox_norm[3] / 1000.0 * height))
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def qwen_predict_bbox_xyxy(processor, model, color_rgb: np.ndarray, vlm_prompt: str, max_new_tokens: int = 128):
    """Ask Qwen for a bbox; return (xyxy_pixels or None, raw_text)."""
    h, w = color_rgb.shape[:2]
    full_prompt = (
        f"{vlm_prompt}. "
        'Return a JSON object with key "bbox_2d" as [x1, y1, x2, y2] '
        "using normalized coordinates from 0 to 1000."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": PILImage.fromarray(color_rgb)},
                {"type": "text", "text": full_prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    input_len = inputs["input_ids"].shape[-1]
    raw_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    bboxes_norm = _parse_qwen_bboxes_norm(raw_text)
    if not bboxes_norm:
        return None, raw_text

    xyxy_list = []
    for bbox_norm in bboxes_norm:
        xyxy = _norm1000_bbox_to_xyxy(bbox_norm, w, h)
        if xyxy is not None:
            xyxy_list.append(xyxy)
    if not xyxy_list:
        return None, raw_text
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in xyxy_list]
    return xyxy_list[int(np.argmax(areas))], raw_text


def _bytes_to_gb(n: int) -> float:
    return n / (1024.0 ** 3)


def report_cuda(label: str):
    """Print allocated / reserved / peak CUDA memory for the current device."""
    if not torch.cuda.is_available():
        print(f"[{label}] CUDA not available")
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    free, total = torch.cuda.mem_get_info()
    print(
        f"[{label}] "
        f"allocated={_bytes_to_gb(allocated):.3f} GiB  "
        f"reserved={_bytes_to_gb(reserved):.3f} GiB  "
        f"peak_alloc={_bytes_to_gb(peak_alloc):.3f} GiB  "
        f"peak_reserved={_bytes_to_gb(peak_reserved):.3f} GiB  "
        f"free={_bytes_to_gb(free):.3f}/{_bytes_to_gb(total):.3f} GiB"
    )


def main(args):
    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"image={image_path}")
    print(f"qwen_model={args.qwen_model}")
    print(f"vlm_prompt={args.vlm_prompt!r}")
    print(f"with_sam2={args.with_sam2}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    report_cuda("baseline")

    pil = PILImage.open(image_path).convert("RGB")
    if args.resize_factor != 1:
        w, h = pil.size
        pil = pil.resize((w // args.resize_factor, h // args.resize_factor), PILImage.BILINEAR)
    color_rgb = np.asarray(pil)
    print(f"image_shape={color_rgb.shape} (HxWxC)")

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    t0 = time.time()
    print(f"Loading processor {args.qwen_model}...")
    processor = AutoProcessor.from_pretrained(args.qwen_model)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading model {args.qwen_model} (dtype={dtype})...")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.qwen_model,
        torch_dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    print(f"Qwen load done in {(time.time() - t0) * 1000.0:.1f} ms")
    report_cuda("after_qwen_load")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    bbox_xyxy, raw_text = qwen_predict_bbox_xyxy(
        processor, model, color_rgb, args.vlm_prompt, max_new_tokens=args.max_new_tokens
    )
    print(f"Qwen infer done in {(time.time() - t0) * 1000.0:.1f} ms")
    print(f"raw_text={raw_text!r}")
    print(f"bbox_xyxy={bbox_xyxy}")
    report_cuda("after_qwen_infer")

    if args.save_vis and bbox_xyxy is not None and not args.with_sam2:
        import cv2

        vis = color_rgb.copy()
        x1, y1, x2, y2 = bbox_xyxy
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        out_path = os.path.abspath(args.save_vis)
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"wrote visualization to {out_path}")

    masks_np = []
    if args.with_sam2:
        from ultralytics.models.sam import SAM2Predictor

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        print(f"Loading SAM2 {args.sam2_model}...")
        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            model=args.sam2_model,
            half=True,
            save=False,
            verbose=False,
        )
        seg_model = SAM2Predictor(overrides=overrides)
        seg_model.set_image(color_rgb)
        box = bbox_xyxy if bbox_xyxy is not None else [100, 100, 300, 400]
        results = seg_model(bboxes=[box])
        n_masks = 0
        if results and getattr(results[0], "masks", None) is not None:
            n_masks = len(results[0].masks)
            masks_np = [results[0].masks[i].data.cpu().numpy()[0] for i in range(n_masks)]
        print(f"SAM2 done in {(time.time() - t0) * 1000.0:.1f} ms, masks={n_masks}, box={box}")
        report_cuda("after_qwen_plus_sam2")

        if args.save_vis:
            import cv2

            vis = color_rgb.copy()
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
            ]
            for i, mask in enumerate(masks_np):
                m = np.asarray(mask)
                if m.ndim == 3:
                    m = m[0]
                if m.shape[:2] != vis.shape[:2]:
                    m = cv2.resize(m.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
                sel = m.astype(bool)
                if not sel.any():
                    continue
                color = np.array(colors[i % len(colors)], dtype=np.float32)
                vis[sel] = (0.45 * vis[sel].astype(np.float32) + 0.55 * color).astype(np.uint8)
            if bbox_xyxy is not None:
                x1, y1, x2, y2 = bbox_xyxy
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            out_path = os.path.abspath(args.save_vis)
            cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"wrote visualization with {len(masks_np)} mask(s) to {out_path}")

    report_cuda("final")
    print("done")


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Probe Qwen VLM GPU memory on a demo image (qwen_sam2 path).")
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join(_here, "illustrations", "objects_hackathon2.jpeg"),
        help="RGB image path.",
    )
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3.5-0.8B", help="Hugging Face id or local path for the Qwen VLM.")
    parser.add_argument("--vlm_prompt", type=str, default="Locate the milk", help="Locate prompt sent to Qwen.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Generation budget for the bbox JSON.")
    parser.add_argument("--resize_factor", type=int, default=1, help="Divide image width/height by this factor before inference.")
    parser.add_argument("--with_sam2", action="store_true", help="Also load SAM2 after Qwen to measure combined memory (qwen_sam2 mode).")
    parser.add_argument("--sam2_model", type=str, default="sam2_s.pt", help="SAM2 checkpoint path (used with --with_sam2).")
    parser.add_argument("--save_vis", type=str, default="", help="Optional path to write RGB image with bbox (and SAM2 mask when --with_sam2).")
    args = parser.parse_args()
    main(args)
