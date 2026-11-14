#!/usr/bin/env python3
"""Score uf_overlap_v1 captures using one global one-to-one label matching.

All retained fragments, including invalid-root rows (region_id=-1), contribute
to denominators. Matching is optimized once on total_count and reused for
visible and occluded scores. No region-size filtering is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


@dataclass
class Counts:
    total: int = 0
    visible: int = 0
    occluded: int = 0

    def add(self, other: "Counts") -> None:
        self.total += other.total
        self.visible += other.visible
        self.occluded += other.occluded


@dataclass
class Capture:
    overlaps: dict[tuple[int, int], Counts]
    metadata: dict[str, str]


def load_capture(path: Path) -> Capture:
    metadata: dict[str, str] = {}
    overlaps: dict[tuple[int, int], Counts] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        def records():
            for line in stream:
                if line.lstrip().startswith("#"):
                    key, sep, value = line.lstrip()[1:].strip().partition("=")
                    if sep:
                        metadata[key.strip()] = value.strip()
                elif line.strip():
                    yield line

        reader = csv.DictReader(records())
        required = {
            "region_id", "object_id", "total_count",
            "visible_count", "occluded_count",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CSV is missing required overlap-count columns")
        for number, row in enumerate(reader, start=2):
            try:
                region = int(row["region_id"])
                obj = int(row["object_id"])
                values = Counts(
                    int(row["total_count"]), int(row["visible_count"]),
                    int(row["occluded_count"]),
                )
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid integer in CSV record {number}") from error
            if region < -1 or region >= 0xFFFFFFFF:
                raise ValueError(f"Invalid region ID in CSV record {number}")
            if obj < 0 or obj > 0xFFFFFFFF:
                raise ValueError(f"Invalid object ID in CSV record {number}")
            if min(values.total, values.visible, values.occluded) < 0:
                raise ValueError(f"Negative count in CSV record {number}")
            if values.total != values.visible + values.occluded:
                raise ValueError(f"Counts do not sum in CSV record {number}")
            if values.total:
                overlaps.setdefault((region, obj), Counts()).add(values)

    total = sum(value.total for value in overlaps.values())
    # The sparse assignment solver uses floating-point edge weights. This bound
    # keeps the integer objective (including its constant offset) exact.
    if total > 2**52 - 1:
        raise ValueError("Capture exceeds the exact integer range of this solver")
    if "retained_fragments" in metadata:
        if int(metadata["retained_fragments"]) != total:
            raise ValueError("CSV total differs from retained_fragments metadata")
    return Capture(overlaps, metadata)


def _score(correct: int, total: int) -> dict:
    return {
        "correct_fragments": correct,
        "total_fragments": total,
        "accuracy_percent": 100.0 * correct / total if total else None,
    }


def analyze(capture: Capture) -> dict:
    overlaps = capture.overlaps
    objects = sorted({obj for _, obj in overlaps})
    regions = sorted({region for region, _ in overlaps if region >= 0})
    object_index = {obj: i for i, obj in enumerate(objects)}
    region_index = {region: i for i, region in enumerate(regions)}

    # Rows are reference objects. Columns are detected regions, followed by one
    # private dummy column per object. Dummies represent unmatched objects.
    # Adding one to every allowed edge makes weights nonzero for SciPy's sparse
    # solver; every full matching has the same number of edges, so the optimal
    # overlap count is unchanged. Sparse storage avoids a regions x objects
    # dense allocation when many tiny regions are present.
    matching: dict[int, int] = {}
    if objects:
        rows: list[int] = []
        columns: list[int] = []
        weights: list[int] = []
        for (region, obj), counts in overlaps.items():
            if region >= 0:
                rows.append(object_index[obj])
                columns.append(region_index[region])
                weights.append(counts.total + 1)
        for index in range(len(objects)):
            rows.append(index)
            columns.append(len(regions) + index)
            weights.append(1)
        graph = coo_matrix(
            (np.asarray(weights, dtype=np.float64), (rows, columns)),
            shape=(len(objects), len(regions) + len(objects)),
        ).tocsr()
        row_ids, column_ids = min_weight_full_bipartite_matching(
            graph, maximize=True
        )
        for row, column in zip(row_ids, column_ids):
            if column < len(regions):
                matching[objects[int(row)]] = regions[int(column)]

    totals = Counts()
    correct = Counts()
    invalid = Counts()
    per_object: dict[int, Counts] = {obj: Counts() for obj in objects}
    for (region, obj), counts in overlaps.items():
        totals.add(counts)
        per_object[obj].add(counts)
        if region == -1:
            invalid.add(counts)
        elif matching.get(obj) == region:
            correct.add(counts)

    object_results = []
    for obj in objects:
        region = matching.get(obj)
        counts = overlaps.get((region, obj), Counts())
        total = per_object[obj]
        object_results.append({
            "object_id": obj,
            "matched_region_id": region,
            "all": _score(counts.total, total.total),
            "visible": _score(counts.visible, total.visible),
            "occluded": _score(counts.occluded, total.occluded),
        })

    warnings = []
    if invalid.total:
        warnings.append(
            f"{invalid.total} fragments have invalid roots; they count as incorrect."
        )
    if capture.metadata.get("allocation_overflow") == "1":
        warnings.append(
            "The node pool overflowed. Scores describe retained fragments only."
        )
    if int(capture.metadata.get("unretained_stored_fragments", "0")) > 0:
        warnings.append(
            "Some stored fragments were not retained. Verify capture coverage "
            "before interpreting visible/occluded scores."
        )
    if not totals.total:
        warnings.append("No retained fragments: accuracy is undefined.")

    return {
        "metric": "one_to_one_matched_fragment_accuracy",
        "matching_basis": "all retained fragments",
        "software": {"numpy": np.__version__, "scipy": scipy.__version__},
        "metadata": capture.metadata,
        "all": _score(correct.total, totals.total),
        "visible": _score(correct.visible, totals.visible),
        "occluded": _score(correct.occluded, totals.occluded),
        "reference_objects": len(objects),
        "detected_regions": len(regions),
        "matched_pairs": len(matching),
        "unmatched_regions": len(regions) - len(matching),
        "invalid_root_fragments": invalid.total,
        "per_object": object_results,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="Exported overlap-count CSV")
    parser.add_argument("--output", type=Path, help="Output JSON filename")
    args = parser.parse_args()
    output = args.output or args.capture.with_suffix(".accuracy.json")
    if output.resolve() == args.capture.resolve():
        parser.error("Output must differ from the input CSV")
    try:
        result = analyze(load_capture(args.capture))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    for label, key in (
        ("All retained", "all"), ("Visible", "visible"), ("Occluded", "occluded")
    ):
        value = result[key]
        percent = value["accuracy_percent"]
        formatted = "N/A" if percent is None else f"{percent:.4f}%"
        print(
            f"{label}: {value['correct_fragments']} / "
            f"{value['total_fragments']} = {formatted}"
        )
    print(
        f"Objects: {result['reference_objects']}; regions: "
        f"{result['detected_regions']}; matched pairs: {result['matched_pairs']}"
    )
    for warning in result["warnings"]:
        print(f"Note: {warning}")
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
