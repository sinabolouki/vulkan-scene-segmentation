#!/usr/bin/env python3
"""Send captured layers to the SAM2 server and save the returned label maps."""

import argparse
import re
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captures", required=True,
                   help="Folder with layer_XX.png (from WSL: /mnt/c/.../build/captures)")
    p.add_argument("--out", default="sam2_results")
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--points-per-side", type=int, default=None)
    args = p.parse_args()

    print("Server:", requests.get(f"{args.url}/health", timeout=5).json())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.captures).glob("layer_*.png"))
    if not files:
        print("No layer_*.png found in", args.captures)
        return

    total_ms = 0.0
    for f in files:
        layer = int(re.search(r"layer_(\d+)", f.stem).group(1))
        params = {"layer": layer, "format": "png"}
        if args.points_per_side:
            params["points_per_side"] = args.points_per_side

        r = requests.post(f"{args.url}/segment", params=params, data=f.read_bytes(),
                          headers={"Content-Type": "image/png"}, timeout=600)
        r.raise_for_status()

        dst = out / f"layer_{layer:02d}_labels.png"
        dst.write_bytes(r.content)

        # Pillow may open 16-bit PNGs as mode "I;16" or "I" -> values are the same
        label = np.asarray(Image.open(dst))
        total_ms += float(r.headers["X-Total-Ms"])
        print(f"layer {layer:02d}: masks={r.headers['X-Num-Masks']:>4}  "
              f"inference={float(r.headers['X-Inference-Ms']):8.1f} ms  max_id={int(label.max())}")

    print(f"All layers: {total_ms / 1000.0:.1f} s server time")


if __name__ == "__main__":
    main()