# Wake-history term root-cause diagnostics (2026-08-24)

Question: at alpha=16 deg (mesh pitched nose-up, flow +x, full U=5 m/s,
rigid membrane, 30-40 pure-aero proposes) the total normal force came out
negative and growing, while the paper reports Cn ~ 0.92-0.95.

## Findings (evidence in session transcript + scripts below)

1. `roj_term_decompose.py` — force decomposition per pressure term:
   - dp_lift1 (steady Kutta-Joukowski): +0.67 -> +0.83, Wagner-like,
     approaching the paper band. HEALTHY.
   - mf2_history (strong-scheme Mf2_vec1): diverges to -6.5 qS with the
     detached-ring wake; -1.5 after restoring the author's connected
     wake chain; still unphysical.
2. `roj_mf2_rows.py` — per-row decomposition: 85% of the mf2 driver sits in
   wake rows 0-5; each row carries Gamma ~ Gamma_total (persistent LEV
   release enters every shed row through the joint TEV relation).
3. `roj_mf2_zero.py` — zeroing only mf2_history: Cn = +0.69 -> +0.83 rising
   (Wagner), consistent with the paper band. All other physics correct.
4. `roj_weak_dp.py` — replacing Mf2_vec1 with the author's weak-scheme
   dp_add = (Gamma - Gamma_old)/dt (calc_fluid_force.m line 51):
   Cn = +0.695 -> transient dip -> +0.94 at step 40. Textbook Wagner
   response onto the paper band.

## Author's MATLAB reference (FSI_by_FEM_and_UVLM/single_sheet)

- `calc_fluid_force.m`: strong scheme constant = (dp_lift1 + Mf2_vec1)n;
  weak scheme constant = (dp_lift1 + dp_add)n with Mf1 = Mf2_1 = 0.
  Scheme selected by `coupling_flag` in calc_fluid_force.m line ~200.
- `generate_wake.m`: the wake is a CONNECTED chain (r_wake_1 of row k =
  r_wake_2 of row k-1; newest row anchored to the moving TE with the
  sheet's structural velocity). The Python port shed detached rectangles;
  restored as the chain re-anchor in q16_flux_v5m_native.propose().
- Mf2 oracle tests feed MATLAB wake states directly, so the Mf2_vec1
  FORMULA was verified, but the wake DYNAMICS/topology were not.

## Production decision

- NativeV5MConfig.wake_history_mode: "material" (author strong, Yamano
  keeps its oracle-verified path) | "bound_rate" (author weak dp_add; all
  other strong blocks retained: dp_lift1, lift2, Mf2_1, Mf1).
- Rojratsirikul runner freezes wake_history_mode="bound_rate".
- Yamano regression with the chained wake: steps 1-7 improved from the
  frozen baseline (step 4: 7.97% -> 5.44%; step 6: 0.47%); step 8 pending.
