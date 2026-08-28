#!/usr/bin/env bash
# Serial formal queue for the unified Rojratsirikul 2011 reproduction
# (HANDOFF_UNIFIED_FRAMEWORK §9 P7/P8).  One GPU: the four cases run
# sequentially with the SAME frozen parameters — no per-case tuning is
# possible or allowed.  Each case writes its own artifacts; a failure in
# one case does not stop the others (their data is diagnostic regardless).
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:platform:platform/warp_vpm
export PFIELD_DEVICE=cuda:0
export FLUXV_GPU_ONLY=1
export FLUXV_DEVICE=cuda:0
export FLUXV_DTYPE=float64
export FLUXV_V5M_FUSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

OUT=artifacts/baselines/fluxv_v5m_rojratsirikul2011_unified_current
mkdir -p "$OUT"

run_case() {
  local case_id="$1"
  local tag="$2"
  echo "=== [queue] $case_id started $(date -Is) ==="
  python platform/warp_vpm/reproduce_rojratsirikul2011_q16_flux_v5m_native.py \
    --case "$case_id" \
    --output "$OUT/${tag}.json" \
    > "$OUT/${tag}.log" 2>&1
  local code=$?
  echo "=== [queue] $case_id finished exit=$code $(date -Is) ==="
}

# P7: A16 primary long run (t* >= 21 by the frozen default).
run_case ROJ11-A16 ROJ11_A16_FULL
# P8: generality, identical frozen parameters.
run_case ROJ11-A17-MODE ROJ11_A17_MODE_FULL
run_case ROJ11-A10 ROJ11_A10_FULL
run_case ROJ11-A23 ROJ11_A23_FULL
echo "=== [queue] all done $(date -Is) ==="
