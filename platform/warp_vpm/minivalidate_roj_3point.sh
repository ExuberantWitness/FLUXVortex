#!/bin/bash
# Minimal 3-point validation (user directive 20260831): one curve, three
# angles (min 5 / mid 16 / max 25), SHORT slices (t*=1.5, 150 steps each)
# with the full G000/E0 evidence chain on.  NO long runs until this basic
# validation is reviewed.
set -u
cd /home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca
export PYTHONPATH=src:platform:platform/warp_vpm
export PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 FLUXV_V5M_FUSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=artifacts/baselines/fluxv_v5m_rojratsirikul2011_a16_fastlane/validation3
mkdir -p $OUT
for CASE in ROJ11-SWEEP-A05 ROJ11-A16 ROJ11-SWEEP-A25; do
  echo "=== $CASE ($(date)) ==="
  python3 -u platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py \
    --case $CASE --max-aero-steps 150 \
    --output $OUT/${CASE}_T15.json
done
echo "3-point validation complete $(date)"
