#!/usr/bin/env python3
"""
sam2_server.py — Local HTTP server wrapping SAM2 automatic mask generation.

Runs inside WSL2 (CUDA). The Vulkan app on Windows sends each PPLL depth layer
as an RGBA PNG; alpha == 0 marks pixels without a fragment in that layer.

Endpoints
  GET  /health
  POST /segment?layer=N&format=json|png
                [&points_per_side=..&points_per_batch=..]
                [&pred_iou_thresh=..&stability_score_thresh=..]
       body = raw PNG bytes

Label map: uint16 per pixel, 0 = no mask, k = mask id k (1-based).
Sampling: at most points_per_side**2 distinct valid pixels, shared across
8-connected valid regions. This setting is now a budget, not a full-image grid.
Batching: points_per_batch controls how many prompts SAM2 processes at once.
"""

import argparse
import base64
import contextlib
import io
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw
from starlette.concurrency import run_in_threadpool

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# config name (resolved inside the sam2 package) -> checkpoint file name
MODEL_CONFIGS = {
    "tiny":      ("configs/sam2.1/sam2.1_hiera_t.yaml",  "sam2.1_hiera_tiny.pt"),
    "small":     ("configs/sam2.1/sam2.1_hiera_s.yaml",  "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large":     ("configs/sam2.1/sam2.1_hiera_l.yaml",  "sam2.1_hiera_large.pt"),
}

MAX_LABELS = 65535  # uint16 label map


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def decode_png(data: bytes):
    """Returns (rgb uint8 HxWx3, valid bool HxW)."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    rgba = np.asarray(img.convert("RGBA"))      # images without alpha -> alpha = 255
    rgb = np.ascontiguousarray(rgba[..., :3])
    valid = rgba[..., 3] > 0                    # alpha 0 = no fragment in this layer
    return rgb, valid


def encode_label_png(label: np.ndarray) -> bytes:
    """uint16 HxW -> 16-bit grayscale PNG bytes."""
    assert label.dtype == np.uint16
    buf = io.BytesIO()
    Image.fromarray(label).save(buf, format="PNG")
    return buf.getvalue()


def _valid_component_ids(valid):
    """8-connected IDs aligned with np.nonzero(valid), using NumPy only.

    Union adjacent row runs instead of visiting every pixel in Python.
    Components describe fragment coverage, not object identities.
    """
    parent, lengths, previous = [], [], []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for row in valid:
        edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        current, j = [], 0
        for start, stop in zip(starts.tolist(), stops.tolist()):
            run_id = len(parent)
            parent.append(run_id)
            lengths.append(stop - start)
            # Half-open runs can connect diagonally at either endpoint.
            while j < len(previous) and previous[j][1] < start:
                j += 1
            k = j
            while k < len(previous) and previous[k][0] <= stop:
                a, b = find(run_id), find(previous[k][2])
                if a != b:
                    parent[max(a, b)] = min(a, b)
                k += 1
            current.append((start, stop, run_id))
        previous = current

    roots = np.asarray([find(i) for i in range(len(parent))], dtype=np.int64)
    _, run_components = np.unique(roots, return_inverse=True)
    return np.repeat(run_components, np.asarray(lengths, dtype=np.int64))


def _spread_points(points_xy, count):
    """Split a region into spatial bins and pick a valid pixel near each center."""
    selected, pending = [], [(points_xy, int(count))]
    while pending:
        points, n = pending.pop()
        if n == len(points):
            selected.append(points)
        elif n == 1:
            distance = np.sum((points - points.mean(axis=0)) ** 2, axis=1)
            selected.append(points[np.argmin(distance)][None, :])
        else:
            axis = 0 if np.ptp(points[:, 0]) >= np.ptp(points[:, 1]) else 1
            ordered = points[np.argsort(points[:, axis], kind="stable")]
            left_count = n // 2
            split = len(points) * left_count // n
            pending.append((ordered[split:], n - left_count))
            pending.append((ordered[:split], left_count))
    return np.concatenate(selected, axis=0)


def build_valid_point_grid(valid, points_per_side):
    """Return normalized XY prompts and a reproducible sampling report.

    Reserve one prompt per connected region, then distribute the remainder
    proportionally to its unused pixels. If regions outnumber the budget,
    prefer larger regions and report how many received no prompt.
    """
    t0 = time.perf_counter()
    valid = np.asarray(valid, dtype=bool)
    h, w = valid.shape
    side = int(points_per_side)
    if side < 1 or h < 1 or w < 1:
        raise ValueError("Sampling requires a nonempty image and positive point budget")
    budget = side * side
    ys, xs = np.nonzero(valid)
    available = np.column_stack((xs, ys))
    count = min(budget, len(available))
    component_ids = _valid_component_ids(valid)
    sizes = np.bincount(component_ids)
    quotas = np.zeros(len(sizes), dtype=np.int64)

    if count and len(sizes) > count:
        quotas[np.argsort(-sizes, kind="stable")[:count]] = 1
    elif count:
        quotas[:] = 1
        remaining = count - len(sizes)
        if remaining:
            capacity = sizes - 1
            extra, remainder = np.divmod(remaining * capacity, capacity.sum())
            quotas += extra
            leftover = int(count - quotas.sum())
            quotas[np.argsort(-remainder, kind="stable")[:leftover]] += 1

    offset = 1.0 / (2 * side)
    old_axis = np.linspace(offset, 1.0 - offset, side)
    old_x = np.minimum((old_axis * w).astype(np.int64), w - 1)
    old_y = np.minimum((old_axis * h).astype(np.int64), h - 1)
    if len(available) == h * w and side <= min(h, w):
        # A fully valid image already permits a cheap, well-spread regular grid.
        grid_x, grid_y = np.meshgrid(old_x, old_y)
        selected = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    elif count == len(available):
        selected = available  # Includes the empty-image case; no duplicates.
    else:
        order = np.argsort(component_ids, kind="stable")
        grouped = available[order]
        offsets = np.concatenate(([0], np.cumsum(sizes)))
        selected = np.concatenate([
            _spread_points(grouped[offsets[i]:offsets[i + 1]], quotas[i])
            for i in np.flatnonzero(quotas)
        ], axis=0)

    # XY order, pixel centers, normalized to the original image dimensions.
    points = (selected.astype(np.float32) + 0.5) / np.array([w, h], np.float32)
    report = {
        "strategy": "valid_components_spatial",
        "connectivity": 8,
        "point_budget": budget,
        "num_points": int(len(points)),
        "valid_pixels": int(len(available)),
        "components_total": int(len(sizes)),
        "components_prompted": int(np.count_nonzero(quotas)),
        "components_without_prompt": int(np.count_nonzero(quotas == 0)),
        "old_grid_points_on_valid": int(valid[np.ix_(old_y, old_x)].sum()),
        "points_outside_valid": int(np.count_nonzero(~valid[selected[:, 1], selected[:, 0]])),
        "points_xy": selected.tolist(),
        "coordinate_convention": "Zero-based pixel indices; SAM2 uses (x+0.5)/width, (y+0.5)/height.",
        "sampling_ms": (time.perf_counter() - t0) * 1000.0,
    }
    return np.ascontiguousarray(points), report


def build_label_map(raw_masks, valid, min_area, min_valid_ratio, diagnostics=None):
    """
    Clip SAM2 masks to the valid (fragment-covered) region, filter them,
    and paint a label map (largest first, so small masks stay on top).
    """
    h, w = valid.shape
    kept = []
    records = []
    for proposal_id, m in enumerate(raw_masks, start=1):
        seg = m["segmentation"]                 # bool HxW (output_mode="binary_mask")
        raw_area = int(m["area"])
        record = {
            "proposal_id": proposal_id,
            "generator_area": raw_area,
            "clipped_area": 0,
            "valid_ratio": None,
            "predicted_iou": float(m["predicted_iou"]),
            "stability_score": float(m["stability_score"]),
            "label_id": 0,
            "visible_area": 0,
            "decision": "empty_generator_mask",
        }
        records.append(record)
        if raw_area == 0:
            continue
        clipped = seg & valid
        area = int(clipped.sum())
        record["clipped_area"] = area
        record["valid_ratio"] = area / raw_area
        # Preserve the existing order and filtering thresholds exactly.
        # Rejection counts use the first failed condition, so they add up.
        if area < min_area:
            record["decision"] = "rejected_min_area"
            continue
        if (area / raw_area) < min_valid_ratio:
            record["decision"] = "rejected_valid_ratio"
            continue
        record["decision"] = "rejected_label_limit"
        kept.append((area, clipped, m, record))

    kept.sort(key=lambda t: t[0], reverse=True)
    kept = kept[:MAX_LABELS]

    label = np.zeros((h, w), dtype=np.uint16)
    for i, (_, seg, _, record) in enumerate(kept, start=1):
        label[seg] = i
        record["label_id"] = i

    visible = np.bincount(label.ravel(), minlength=len(kept) + 1)

    meta = []
    for i, (area, seg, m, record) in enumerate(kept, start=1):
        record["visible_area"] = int(visible[i])
        record["decision"] = "kept_visible" if visible[i] > 0 else "fully_overwritten"
        ys, xs = np.nonzero(seg)
        meta.append({
            "id": i,
            "area": area,                        # clipped area
            "visible_area": int(visible[i]),     # pixels that kept this id after overlap
            "bbox": [int(xs.min()), int(ys.min()),
                     int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],  # x,y,w,h
            "predicted_iou": float(m["predicted_iou"]),
            "stability_score": float(m["stability_score"]),
        })
    if diagnostics is not None:
        valid_pixels = int(valid.sum())
        labeled_valid = int(((label > 0) & valid).sum())
        reasons = ("empty_generator_mask", "rejected_min_area",
                   "rejected_valid_ratio", "rejected_label_limit",
                   "fully_overwritten", "kept_visible")
        counts = {reason: sum(r["decision"] == reason for r in records)
                  for reason in reasons}
        counts.update({
            "generator_masks": len(raw_masks),
            "kept_before_overlap": len(meta),
            "surviving_labels": counts["kept_visible"],
            "valid_pixels": valid_pixels,
            "labeled_valid_pixels": labeled_valid,
            "unlabeled_valid_pixels": valid_pixels - labeled_valid,
            "labels_outside_valid": int(((label > 0) & ~valid).sum()),
            "valid_coverage": labeled_valid / valid_pixels if valid_pixels else None,
        })
        diagnostics.update({"counts": counts, "proposals": records})
    return label, meta


def make_mask_image(label):
    """Color the final masks on black; label 0 is the unlabeled background."""
    n = int(label.max())
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 256, size=(n + 1, 3), dtype=np.uint8)
    colors[0] = 0
    return colors[label]


def make_overlay(rgb, label, alpha=0.45):
    mask_image = make_mask_image(label)
    out = rgb.astype(np.float32)
    mask = label > 0
    out[mask] = (1 - alpha) * out[mask] + alpha * mask_image[mask].astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def parse_size(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def save_diagnostics(root, rgb, valid, label, raw_masks, meta, diagnostics,
                     layer, model_name, params, min_area, min_valid_ratio,
                     inference_ms, processing_ms):
    """Save one unique request; proposal PNGs preserve overlaps and rejected masks."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    request_dir = root / f"{stamp}_layer_{layer:02d}_{uuid4().hex[:8]}"
    masks_dir = request_dir / "generator_masks"
    masks_dir.mkdir(parents=True, exist_ok=False)

    Image.fromarray(rgb).save(request_dir / "input_rgb.png")
    Image.fromarray(valid.astype(np.uint8) * 255).save(request_dir / "valid.png")
    Image.fromarray(label).save(request_dir / "labels.png")
    Image.fromarray(make_mask_image(label)).save(request_dir / "masks.png")
    Image.fromarray(make_overlay(rgb, label)).save(request_dir / "overlay.png")
    prompts = Image.fromarray(rgb)
    draw = ImageDraw.Draw(prompts)
    for x, y in diagnostics["sampling"]["points_xy"]:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 255), outline=(0, 0, 0))
        draw.point((x, y), fill=(255, 255, 0))
    prompts.save(request_dir / "prompts.png")
    for proposal_id, m in enumerate(raw_masks, start=1):
        mask = np.asarray(m["segmentation"], dtype=np.uint8) * 255
        Image.fromarray(mask).save(masks_dir / f"proposal_{proposal_id:06d}.png")

    report = {
        "schema_version": 1,
        "layer": layer,
        "width": int(valid.shape[1]), "height": int(valid.shape[0]),
        "model": model_name,
        "params": params,
        "filters": {"min_area": min_area, "min_valid_ratio": min_valid_ratio},
        "timing_ms": {"inference": inference_ms,
                      "sampling": diagnostics["sampling"]["sampling_ms"],
                      "processing_before_debug_io": processing_ms},
        "notes": [
            "Generator proposals are AFTER SAM2's internal filtering/NMS.",
            "Rejection counts use the first failed condition: area, then valid ratio.",
            "The label map keeps the original largest-first, small-masks-on-top policy.",
            "masks.png shows the final labels in the overlay's colors on a black background.",
            "proposal_id identifies a generator_masks PNG; label_id is the returned ID.",
            "HTTP total time includes diagnostic writes; inference time does not.",
            "prompts.png marks sampled pixel centers in yellow; cyan markers extend beyond them.",
            "Connected valid regions are sampling regions, not object identities.",
        ],
        **diagnostics,
        "returned_mask_metadata": meta,
    }
    (request_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return request_dir


# ─────────────────────────────────────────────────────────────────────────────
# SAM2 service (model loaded once, one inference at a time)
# ─────────────────────────────────────────────────────────────────────────────

class Sam2Service:
    def __init__(self, args):
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in this environment")
        self.device = torch.device(args.device)
        self.is_cuda = self.device.type == "cuda"

        cfg_name, ckpt_name = MODEL_CONFIGS[args.model]
        ckpt = Path(args.checkpoint_dir).expanduser() / ckpt_name
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        if self.is_cuda and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True   # Ampere+ (as in SAM2 examples)
            torch.backends.cudnn.allow_tf32 = True

        print(f"[sam2] Loading '{args.model}' from {ckpt} on {self.device} ...")
        self.model = build_sam2(cfg_name, str(ckpt), device=self.device,
                                apply_postprocessing=args.apply_postprocessing)
        self.model_name = args.model

        self.use_bf16 = self.is_cuda and not args.no_bf16 and torch.cuda.is_bf16_supported()

        self.defaults = {
            "points_per_side": args.points_per_side,
            "points_per_batch": args.points_per_batch,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_score_thresh,
        }
        print(f"[sam2] Prompt batch size: {args.points_per_batch}")
        self.min_area = args.min_area
        self.min_valid_ratio = args.min_valid_ratio

        self.debug_dir = None
        if args.debug_dir:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            self.debug_dir = (Path(args.debug_dir).expanduser().resolve()
                              / f"run_{stamp}_{uuid4().hex[:8]}")
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=False)
            print(f"[sam2] Diagnostics run directory: {self.debug_dir}")

        self._generators = {}          # cache: param tuple -> SAM2AutomaticMaskGenerator
        self.lock = threading.Lock()   # the GPU model is not safe for concurrent calls
        self.warmup_info = None

    # ---- internals (call with self.lock held) -------------------------------

    def _get_generator(self, params, point_grid):
        key = (params["points_per_side"], params["pred_iou_thresh"],
               params["stability_score_thresh"], params["points_per_batch"])
        gen = self._generators.get(key)
        if gen is None:
            gen = SAM2AutomaticMaskGenerator(
                self.model,
                points_per_side=None,
                points_per_batch=key[3],
                point_grids=[point_grid],
                crop_n_layers=0,       # This grid uses the full image coordinates.
                pred_iou_thresh=key[1],
                stability_score_thresh=key[2],
                output_mode="binary_mask",
            )
            self._generators[key] = gen
        # Cached generators must receive THIS request's validity-based prompts.
        # Updating the grid and generating masks both happen under self.lock.
        gen.point_grids = [point_grid]
        return gen

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _generate(self, rgb, params, point_grid):
        gen = self._get_generator(params, point_grid)
        if self.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), self._autocast():
            raw = gen.generate(rgb)
        if self.is_cuda:
            torch.cuda.synchronize()
        return raw, (time.perf_counter() - t0) * 1000.0

    # ---- public --------------------------------------------------------------

    def warmup(self, width, height, runs):
        if runs <= 0:
            return
        # Smooth gradient instead of noise: noise would produce huge mask counts.
        x = np.linspace(0, 255, width, dtype=np.float32)
        y = np.linspace(0, 255, height, dtype=np.float32)
        img = np.stack(np.meshgrid(x, y), axis=-1)
        img = np.concatenate([img, np.full((height, width, 1), 128, np.float32)], axis=-1)
        img = img.astype(np.uint8)
        point_grid, _ = build_valid_point_grid(
            np.ones((height, width), dtype=bool), self.defaults["points_per_side"])

        print(f"[sam2] Warm-up: {runs} run(s) at {width}x{height} ...")
        times = []
        with self.lock:
            for _ in range(runs):
                _, ms = self._generate(img, self.defaults, point_grid)
                times.append(ms)
        self.warmup_info = {"size": [width, height], "runs_ms": [round(t, 1) for t in times]}
        print(f"[sam2] Warm-up done: {', '.join(f'{t:.0f} ms' for t in times)}")

    def run(self, rgb, valid, layer, params):
        t_total = time.perf_counter()
        h, w = valid.shape

        point_grid, sampling = build_valid_point_grid(valid, params["points_per_side"])
        raw, inference_ms = [], 0.0
        if len(point_grid):            # Empty grids must not reach SAM2.
            with self.lock:
                raw, inference_ms = self._generate(rgb, params, point_grid)

        diagnostics = {"sampling": sampling}
        label, meta = build_label_map(raw, valid, self.min_area,
                                     self.min_valid_ratio, diagnostics)

        if self.debug_dir is not None:
            try:
                saved = save_diagnostics(
                    self.debug_dir, rgb, valid, label, raw, meta, diagnostics,
                    layer, self.model_name, params, self.min_area, self.min_valid_ratio,
                    inference_ms, (time.perf_counter() - t_total) * 1000.0)
                print(f"[sam2] Diagnostics: {saved}")
            except OSError as e:
                print(f"[sam2] Could not save diagnostics: {e}")

        total_ms = (time.perf_counter() - t_total) * 1000.0
        c = diagnostics["counts"]
        coverage = (f"{100 * c['valid_coverage']:.1f}%"
                    if c["valid_coverage"] is not None else "n/a (empty layer)")
        print(f"[sam2] layer={layer:>3}  {w}x{h}  valid={valid.mean() * 100:5.1f}%  "
              f"prompts={sampling['num_points']}  "
              f"batch_size={params['points_per_batch']}  "
              f"old_grid_valid={sampling['old_grid_points_on_valid']}  "
              f"regions={sampling['components_prompted']}/{sampling['components_total']}  "
              f"generator={len(raw)}  rejected_area={c['rejected_min_area']}  "
              f"rejected_ratio={c['rejected_valid_ratio']}  "
              f"kept={len(meta)}  overwritten={c['fully_overwritten']}  "
              f"surviving={c['surviving_labels']}  valid_coverage={coverage}  "
              f"inference={inference_ms:8.1f} ms  total={total_ms:8.1f} ms")
        return label, meta, inference_ms, total_ms


# ─────────────────────────────────────────────────────────────────────────────
# HTTP API
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="SAM2 layer segmentation server")
SERVICE: Optional[Sam2Service] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": SERVICE.model_name,
        "device": str(SERVICE.device),
        "gpu": torch.cuda.get_device_name(0) if SERVICE.is_cuda else None,
        "bf16": SERVICE.use_bf16,
        "defaults": SERVICE.defaults,
        "min_area": SERVICE.min_area,
        "min_valid_ratio": SERVICE.min_valid_ratio,
        "warmup": SERVICE.warmup_info,
    }


@app.post("/segment")
async def segment(
    request: Request,
    layer: int = Query(-1),
    format: str = Query("json"),
    points_per_side: Optional[int] = Query(None, ge=4, le=128),
    points_per_batch: Optional[int] = Query(None, ge=1),
    pred_iou_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
    stability_score_thresh: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    if format not in ("json", "png"):
        raise HTTPException(status_code=400, detail="format must be 'json' or 'png'")

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body (expected PNG bytes)")

    rgb, valid = decode_png(body)

    params = dict(SERVICE.defaults)
    if points_per_side is not None:        params["points_per_side"] = points_per_side
    if points_per_batch is not None:       params["points_per_batch"] = points_per_batch
    if pred_iou_thresh is not None:        params["pred_iou_thresh"] = pred_iou_thresh
    if stability_score_thresh is not None: params["stability_score_thresh"] = stability_score_thresh

    # Blocking GPU work in a worker thread -> /health stays responsive during inference
    label, meta, inference_ms, total_ms = await run_in_threadpool(
        SERVICE.run, rgb, valid, layer, params)

    png_bytes = encode_label_png(label)
    h, w = label.shape

    if format == "png":
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "X-Num-Masks": str(len(meta)),
                "X-Width": str(w),
                "X-Height": str(h),
                "X-Layer": str(layer),
                "X-Inference-Ms": f"{inference_ms:.2f}",
                "X-Total-Ms": f"{total_ms:.2f}",
            },
        )

    return JSONResponse({
        "layer": layer,
        "width": w,
        "height": h,
        "num_masks": len(meta),
        "inference_ms": inference_ms,
        "total_ms": total_ms,
        "params": params,
        "masks": meta,
        "label_map_png": base64.b64encode(png_bytes).decode("ascii"),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global SERVICE
    p = argparse.ArgumentParser(description="SAM2 layer segmentation server")
    p.add_argument("--host", default="0.0.0.0",
                   help="0.0.0.0 so the Windows host can reach the server inside WSL2")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", choices=MODEL_CONFIGS.keys(), default="base_plus")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Folder with sam2.1_hiera_*.pt, e.g. ~/sam2/checkpoints")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--apply-postprocessing", action="store_true",
                   help="build_sam2 default (your old test script); "
                        "the official AMG notebook uses False")
    p.add_argument("--points-per-side", type=int, default=32,
                   help="Prompt budget is N*N, sampled only inside valid pixels (default: 1024)")
    p.add_argument("--points-per-batch", type=int, default=64,
                   help="Prompts processed at once (default: 64); "
                        "higher values may be faster but use more GPU memory")
    p.add_argument("--pred-iou-thresh", type=float, default=0.8)
    p.add_argument("--stability-score-thresh", type=float, default=0.90)
    p.add_argument("--min-area", type=int, default=64,
                   help="Drop masks with fewer valid pixels than this")
    p.add_argument("--min-valid-ratio", type=float, default=0.5,
                   help="Drop masks lying mostly on alpha=0 (empty) pixels")
    p.add_argument("--warmup-size", default="1280x720",
                   help="Use your capture resolution (WxH)")
    p.add_argument("--warmup-runs", type=int, default=2)
    p.add_argument("--debug-dir", default="",
                   help="If set: saves RGB, validity, individual generator masks, labels, "
                        "colored masks, overlays, prompt locations and reports in unique run/request subdirectories")
    args = p.parse_args()
    if args.points_per_batch < 1:
        p.error("--points-per-batch must be at least 1")

    SERVICE = Sam2Service(args)
    SERVICE.warmup(*parse_size(args.warmup_size), args.warmup_runs)
    print(f"[sam2] Ready on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
