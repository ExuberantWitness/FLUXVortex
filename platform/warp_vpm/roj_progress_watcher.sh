#!/bin/bash
# Per-figure incremental progress: whenever a new finished payload or rigid
# case JSON appears, refresh all comparison figures and append the MAE
# lines to comparison/progress.log.  Lets the human stop the queue at any
# figure boundary with a full picture.
cd /home/exuber/CODE/CORE/pythonProject1/AUTORESEARCH/FLUXV_RUNS/v5m-fa8eaca
export PYTHONPATH=src:platform:platform/warp_vpm
OUT=artifacts/baselines/fluxv_v5m_rojratsirikul2011_fig06_09_12_15_unified_current
LAST=""
while true; do
  COUNT=$(ls $OUT/membrane_sweep/ROJ11-*.json 2>/dev/null | grep -v partial | wc -l)
  RIGID=$(ls $OUT/cases/*.json 2>/dev/null | wc -l)
  SIG="$COUNT-$RIGID"
  if [ "$SIG" != "$LAST" ]; then
    echo "[$(date '+%H:%M')] membrane payloads=$COUNT rigid cases=$RIGID" >> $OUT/comparison/progress.log
    python3 platform/warp_vpm/compare_rojratsirikul2011_digitized_oracles.py \
      >> $OUT/comparison/progress.log 2>&1
    grep -E "H[0-9]" $OUT/comparison/progress.log | tail -6 >> /dev/null
    LAST="$SIG"
  fi
  sleep 600
done
