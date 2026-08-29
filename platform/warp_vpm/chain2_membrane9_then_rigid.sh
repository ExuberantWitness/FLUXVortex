#!/bin/bash
# 9-point membrane Figure 6/9 curve (after the in-flight A10) then the
# rigid U=5 curve.  Sequential, checkpoint-resumable, watcher refreshes
# figures after every case.
set -u
cd /home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca
export PYTHONPATH=src:platform:platform/warp_vpm
export PFIELD_DEVICE=cuda:0 FLUXV_GPU_ONLY=1 FLUXV_V5M_FUSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=artifacts/baselines/fluxv_v5m_rojratsirikul2011_fig06_09_12_15_unified_current
mkdir -p $OUT/membrane_sweep

# Wait for the in-flight A10 python (owned by the retired chain wrapper).
while pgrep -f "reproduce_rojratsirikul2011_q16_flux_v5m_native.py --case ROJ11-A10" > /dev/null; do
  sleep 60
done
echo "=== A10 finished, starting extended sweep ($(date)) ==="

for CASE in ROJ11-SWEEP-A21 ROJ11-SWEEP-A13 ROJ11-SWEEP-A19 ROJ11-A23 ROJ11-SWEEP-A25; do
  if [ -f $OUT/membrane_sweep/${CASE}_T10.json ]; then
    echo "=== skip $CASE (payload exists) ==="
    continue
  fi
  echo "=== membrane $CASE ($(date)) ==="
  python3 -u platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py \
    --case $CASE --max-aero-steps 1000 \
    --output $OUT/membrane_sweep/${CASE}_T10.json
done
echo "=== rigid U5 curve ($(date)) ==="
python3 -u platform/warp_vpm/queue_roj_rigid_fig9_12_13_15.py
echo "chain2 complete $(date)"
