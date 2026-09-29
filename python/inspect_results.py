#!/usr/bin/env python3
"""Inspect SAM2 diagnostic requests or the original flat capture/result folders.

New diagnostics:
  python inspect_results.py --results sam2_debug/run_... --out inspect_out

Original flat outputs:
  python inspect_results.py --captures captures --results old_debug --out inspect_out

No model or GPU is required. Requires numpy and Pillow.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def colorize(label):
    rng = np.random.default_rng(0)
    colors = rng.integers(40, 256, size=(int(label.max()) + 1, 3), dtype=np.uint8)
    colors[0] = 0
    return colors[label]


def inspect_input(rgb, valid):
    print(f"  input: {rgb.shape[1]}x{rgb.shape[0]}  valid={valid.mean() * 100:.1f}%")
    if not valid.any():
        print("    Empty layer: there are no target fragments to label.")
    else:
        pixels = rgb[valid].astype(np.float32)
        spatial_std = pixels.std(axis=0)
        print(f"    valid RGB mean={pixels.mean(axis=0).round(1).tolist()}  "
              f"spatial std per channel={spatial_std.round(1).tolist()}")
        if spatial_std.max() < 1.0:
            print("    Low color variation; this alone does not determine SAM2's result.")


def inspect_label(path, rgb, valid, out_dir):
    with Image.open(path) as image:
        mode = image.mode
        label = np.asarray(image).copy()
    if label.ndim != 2 or not np.issubdtype(label.dtype, np.integer):
        raise ValueError(f"Expected an integer grayscale label map: {path}")
    if label.shape != valid.shape or rgb.shape[:2] != valid.shape:
        raise ValueError(f"RGB, validity and label dimensions differ: {path}")
    if label.min() < 0 or label.max() > 65535:
        raise ValueError(f"Labels are outside uint16 range: {path}")

    label = label.astype(np.uint16)
    labeled = label > 0
    ids = np.unique(label[labeled])
    valid_pixels = int(valid.sum())
    labeled_valid = int((labeled & valid).sum())
    outside = int((labeled & ~valid).sum())
    coverage = f"{100 * labeled_valid / valid_pixels:.1f}%" if valid_pixels else "n/a"
    print(f"  labels: mode={mode}  masks present={len(ids)}  max ID={label.max()}")
    print(f"    valid coverage={coverage} ({labeled_valid}/{valid_pixels})  "
          f"unlabeled valid={valid_pixels - labeled_valid}  labels outside valid={outside}")
    if not len(ids):
        print("    Final label map is empty; inspect proposal rejection reasons.")
    if outside:
        print("    ERROR: labels exist outside the target layer's valid pixels.")

    out_dir.mkdir(parents=True, exist_ok=True)
    color = colorize(label)
    Image.fromarray(rgb).save(out_dir / "input_rgb.png")
    Image.fromarray(valid.astype(np.uint8) * 255).save(out_dir / "valid.png")
    Image.fromarray(color).save(out_dir / "labels_color.png")
    overlay = rgb.astype(np.float32)
    overlay[labeled] = 0.55 * overlay[labeled] + 0.45 * color[labeled].astype(np.float32)
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(out_dir / "overlay.png")
    Image.fromarray((valid & ~labeled).astype(np.uint8) * 255).save(
        out_dir / "unlabeled_valid.png")
    return len(ids), outside


def inspect_request(report_path, out_dir):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    folder = report_path.parent
    with Image.open(folder / "input_rgb.png") as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    with Image.open(folder / "valid.png") as image:
        valid = np.asarray(image.convert("L")) > 0
    print(f"Layer {report['layer']}: {folder}")
    inspect_input(rgb, valid)
    count, outside = inspect_label(folder / "labels.png", rgb, valid, out_dir)
    c = report["counts"]
    print(f"  generator={c['generator_masks']}  rejected_area={c['rejected_min_area']}  "
          f"rejected_ratio={c['rejected_valid_ratio']}  "
          f"empty_proposals={c['empty_generator_mask']}  "
          f"rejected_label_limit={c['rejected_label_limit']}")
    print(f"  kept before overlap={c['kept_before_overlap']}  "
          f"fully overwritten={c['fully_overwritten']}  "
          f"surviving labels={c['surviving_labels']}")
    if count != c["surviving_labels"] or outside != c["labels_outside_valid"]:
        raise ValueError(f"Saved report and label image disagree: {folder}")
    print("  Each generator_masks/proposal_NNNNNN.png is a separate binary proposal.")
    print("  report.json maps proposal_id to label_id and its filtering outcome.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True,
                        help="diagnostics run/request directory, or original flat result folder")
    parser.add_argument("--captures", help="original captures (required only for flat results)")
    parser.add_argument("--out", default="inspect_out")
    args = parser.parse_args()
    results, out = Path(args.results), Path(args.out)
    if not results.is_dir():
        parser.error(f"Results directory does not exist: {results}")

    reports = sorted(results.rglob("report.json"))
    if reports:
        for report_path in reports:
            relative = report_path.parent.relative_to(results)
            inspect_request(report_path, out / relative)
    else:
        if not args.captures:
            parser.error("No report.json found. For original flat results, supply --captures.")
        captures = Path(args.captures)
        if not captures.is_dir():
            parser.error(f"Captures directory does not exist: {captures}")
        matched = 0
        for path in sorted(captures.glob("layer_*.png")):
            match = re.fullmatch(r"layer_(\d+)", path.stem)
            if match is None:
                continue
            matched += 1
            layer = int(match.group(1))
            print(f"Layer {layer:02d}: {path}")
            with Image.open(path) as image:
                rgba = np.asarray(image.convert("RGBA")).copy()
            rgb, valid = rgba[..., :3], rgba[..., 3] > 0
            inspect_input(rgb, valid)
            labels = results / f"layer_{layer:02d}_labels.png"
            if labels.is_file():
                inspect_label(labels, rgb, valid, out / f"layer_{layer:02d}")
            else:
                print(f"  Label file not found: {labels.name}")
        if matched == 0:
            parser.error(f"No layer_XX.png captures found in {captures}")
    print(f"\nInspection images written to: {out.resolve()}")


if __name__ == "__main__":
    main()
