#!/bin/bash
# Run TreeSegPy inference on a list of held-out plots.
# Usage: bash scripts/run_inference_batch.sh
#
# Optional environment overrides:
#   TREESEGPY_IN_DIR   input directory of normalized LAZ files
#   TREESEGPY_OUT_DIR  output directory for predictions
#   TREESEGPY_CKPT     path to the model checkpoint

cd "$(dirname "$0")/.."

CKPT="${TREESEGPY_CKPT:-logs/cw_balanced/best_f1.pt}"
IN_DIR="${TREESEGPY_IN_DIR:-/mnt/d/TopTreeSegR/normalized_clean}"
OUT_DIR="${TREESEGPY_OUT_DIR:-/mnt/d/TopTreeSegR/preds}"

mkdir -p "$OUT_DIR"

PLOTS=(
  "Rem_Herby_2016_2003206"
  "Rem_Katrynka_2016_3702205"
  "Rem_Milicz_2015_2804101"
  "Rem_Suprasl_2015_2402904"
  "Rem_Piensk_2016_0901202"
)

for p in "${PLOTS[@]}"; do
  echo "============================================================"
  echo "plot: $p"
  echo "============================================================"
  if [ -f "$OUT_DIR/${p}.laz" ]; then
    echo "  already done, skipping inference"
    continue
  fi
  python -m treesegpy.predict \
    --input "$IN_DIR/${p}.laz" \
    --checkpoint "$CKPT" \
    --output "$OUT_DIR/${p}.laz" \
    --threshold 0.10 \
    --chunk-size 50000 2>&1 | tee "$OUT_DIR/${p}.log"
done

echo
echo "Running threshold sweep on each output..."
for p in "${PLOTS[@]}"; do
  if [ ! -f "$OUT_DIR/${p}.laz" ]; then
    echo "  missing: $OUT_DIR/${p}.laz, skipping sweep"
    continue
  fi
  python scripts/threshold_sweep.py \
    "$OUT_DIR/${p}.laz" \
    "$IN_DIR/${p}.laz"
done

echo
echo "Done. Threshold sweeps written next to each prediction LAZ."