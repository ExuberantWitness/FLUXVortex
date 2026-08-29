#!/bin/bash
# Membrane Figure 6/9 sweep (t*=10) then rigid queue resume (checkpointed).
set -u
cd /home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca
export PYTHONPATH=src:platform:platform/warp_vpm
export PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 FLUXV_V5M_FUSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=artifacts/baselines/fluxv_v5m_rojratsirikul2011_fig06_09_12_15_unified_current
mkdir -p $OUT/membrane_sweep
for CASE in ROJ11-SWEEP-A05 ROJ11-A10 ROJ11-SWEEP-A21 ROJ11-SWEEP-A25; do
  echo "=== membrane $CASE ($(date)) ==="
  python3 -u platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py \
    --case $CASE --max-aero-steps 1000 \
    --output $OUT/membrane_sweep/${CASE}_T10.json
done
echo "=== rigid queue resume ($(date)) ==="
python3 -u platform/warp_vpm/queue_roj_rigid_fig9_12_13_15.py
echo "chain complete $(date)"
