# Speed engineering plan (approved, in execution order) — continue here

Baseline honest accounting: engineering-only speedup so far ~1.8x (dense-GEMV
transfers 1.27x on top of Aitken persistence); the 5x claim earlier conflated
the 50->10 substep protocol change. Current: 15.7 s/step light, ~26 s/step at
16k particles (particle O(N^2) term dominates growth).

## Item 1 (IN PROGRESS): fused particle Biot-Savart kernel
File: platform/warp_vpm/pfield_torch_gpu.py, method velocity_at_cuda
(lines ~244-276): currently ~15 torch ops per (1024x4096) tile, full
intermediate materialization. Replace with ONE warp kernel per call:
- Math (exact, Gaussian-regularized BS): per source s: delta=t-p[s];
  r2=dot(delta,delta); if r2>0: r=sqrt(r2); rho=r/sigma[s];
  reg=erf(rho*0.7071067811865476) - 0.7978845608028654*rho*exp(-0.5*rho*rho);
  w=-0.07957747154594767*reg/(r2*r); acc += w*cross(delta,gamma[s]).
  Constants: -1/(4pi), sqrt(2/pi), 1/sqrt(2).
- Layout per user's small-particle-computing reference: one WARP (32 thr) per
  target, strided source loop + in-warp reduction (wp.tile_sum or manual
  shuffle), NOT one-thread-serial (that would be ~2e10 flops serialized).
  Fallback if warp-cooperative reductions are awkward: 2D kernel
  (target, source_block) writing partials + tiny second-pass reduce.
- Validation: velocity_at_cuda_fused vs current chunked path on random
  fields, max-abs <= 1e-12; then A/B 30-step probe (dCn <= 1e-9) + full
  pytest; measure s/step at 26k particles (target: particle segment
  8-12 s -> 1-2 s).

## Item 2: CUDA graph of the pure-torch velocity_force path
velocity_force is now warp-free (dense transfers). Capture the four
velocity_force evaluations per substep cluster (fixed shapes) with standard
torch.cuda.graph (warmup on side stream, static input buffers, copy-in/
copy-out). Bit-identity gate vs eager; expected 3-5 s/step saved.
Note: keep kernels_q16_transfer._CAPTURING=False (not needed anymore on
this path).

## Item 3: IQN-ILS coupling accelerator (~150 lines)
Replace/complement _Aitken in q16_flux_v5m_native_fsi.py advance():
reuse last ~10 outer steps of interface residual/displacement pairs,
least-squares secant update (Degroote 2009 IQN-ILS; QR2 filter eps 1e-2,
pre-scale force/displacement blocks). Same fixed point, same 5e-7 tolerance.
Opt-in flag; Yamano default unchanged. Gate: converged residuals equal
within tolerance + iterations drop 3->2 on the A16 probe.

## Queue after engineering
E=1.4 branch running (do not touch; verdict on completion: asymptote camber
->0.043 AND Cn slightly up = double-sided check), then A10/A23 generality
runs at the new speed, same parameters, no per-case tuning.

## Execution record 2026-08-26
- Item 1 DONE (dc04e10): fused B-S kernel, 8.5e-14 vs chunked, 3.2x op.
- Item 2 PARTIAL (944080d): aic LU caching landed (solves 80x/step -> 1
  factorization); gates capture-deferred; graph capture still blocked by an
  unidentified illegal call on the torch path (lu_solve? cusolver internals).
- Item 3 PARKED: naive IQN-ILS DIVERGES on this interface - residual doubles
  per iteration (spectral radius > 1 under added mass); Davis 2022 requires
  force/displacement block PRE-SCALING which we did not implement. Aitken
  (persistent) remains production. Revive only with pre-scaling + QR3.
