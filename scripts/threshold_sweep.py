"""Compute threshold sweep metrics for one plot and append to CSV."""
import csv
from pathlib import Path
import sys

import laspy
import numpy as np


THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]


def sweep(pred_path, gt_path):
    pred = laspy.read(pred_path)
    prob = np.asarray(pred.tree_prob)

    gt = laspy.read(gt_path)
    tid = np.asarray(gt.treeID)
    true_tree = (tid != 0)

    if len(prob) != len(tid):
        raise ValueError(f"point count mismatch: {len(prob)} vs {len(tid)}")

    rows = []
    coverage = 1.0  # inference always produces full coverage in current version
    true_frac = float(true_tree.mean())

    for t in THRESHOLDS:
        pred_tree = prob >= t
        tp = int((pred_tree & true_tree).sum())
        fp = int((pred_tree & ~true_tree).sum())
        fn = int((~pred_tree & true_tree).sum())
        tn = int((~pred_tree & ~true_tree).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        frac = float(pred_tree.mean())
        rows.append({
            "plot": pred_path.stem,
            "threshold": t,
            "precision": round(p, 6),
            "recall": round(r, 6),
            "f1": round(f1, 6),
            "tree_fraction": round(frac, 6),
            "true_fraction": round(true_frac, 6),
            "coverage": round(coverage, 6),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return rows


def main():
    if len(sys.argv) not in (3, 4):
        print("usage: threshold_sweep.py PRED_LAZ GT_LAZ [CSV_PATH]")
        print()
        print("  PRED_LAZ  prediction LAZ produced by treesegpy.predict")
        print("  GT_LAZ    reference LAZ with a treeID field")
        print("  CSV_PATH  optional output CSV (default: alongside PRED_LAZ)")
        sys.exit(1)

    pred_path = Path(sys.argv[1])
    gt_path = Path(sys.argv[2])

    if len(sys.argv) == 4:
        csv_path = Path(sys.argv[3])
    else:
        csv_path = pred_path.with_name("threshold_sweep.csv")

    rows = sweep(pred_path, gt_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"{pred_path.stem}: wrote {len(rows)} rows to {csv_path}")

    # Print compact summary
    print(f"  {'thresh':>6} {'prec':>7} {'rec':>7} {'F1':>7} {'frac':>7}")
    for r in rows:
        print(f"  {r['threshold']:>6.2f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} "
              f"{r['f1']:>7.4f} {r['tree_fraction']:>7.4f}")


if __name__ == "__main__":
    main()