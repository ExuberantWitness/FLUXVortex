# V5H13 checklist

Stage: `experiment`; CONCLUDED 2026-08-17: implementation + oracle + review +
re-sign + graded smoke all PASS; formal A STOPped at the first coarse step
after the frozen k=5 birth window (layer 3, substep 21) — the transient
outlasts the window. The preregistered prediction PASSED exactly (0.1243).
(r=4,k=5) stays frozen; formal B forbidden; next step requires a NEW
preregistered branch (extend k, or Idea C sigma scheduling).

## Governance
- [x] PLAN.md / FREEZE_INPUTS.json preregistered (r=4, k=5 frozen; gate
      accounting amendment frozen; predictions frozen)
- [x] Idea B approved at research-pipeline Gate 1 (Idea A falsified)
- [x] Rehash frozen controls immediately before implementation (all 8 MATCH)

## Implementation (tests-first)
- [x] Mechanical 8-file fork into V5H13 namespace; namespace-normalized diff
      shows only labels before behavioral edits (coupling diff = 0 lines mod
      label swaps; stream = pure byte-copy; forked suites 33/51/11/25 all
      PASS fresh-process; frozen controls no drift)
- [x] Stream schedule parameter + per-record dt binding (record schema gains hashed `substep_delta_time`; 26 tests incl. graded positive/negative matrix)
- [x] Coupling schedule construction + per-substep gate via `view.substep_delta_time` (amendment); `birth_window_refinement` config; 12 tests
- [x] Executor per-record dt, effective FORMAL_LEVELS (47/79/143), nominal mapping, smoke counters (141/282/846/1128); 33 tests
- [x] Runner N_eff tables, per-substep dt cross-check, final coordinate 143, full-matrix 2421 stages; 51 tests
- [ ] Replay determinism test (byte-identical hashes/counts) [stream registry identity test covers in-process determinism; cross-process determinism to be verified at smoke]

## Validation
- [ ] Analytic oracle: order >= 2 on graded grid
- [ ] Frozen focused suites (V5H13 six + per-file fresh-process)
- [ ] Static gates (py_compile/Black/Ruff/diff-check)
- [ ] Fresh hostile review of the final diff
- [ ] Fresh dependency manifest/token; old tokens fail closed
- [ ] Disposable N32/layer1 smoke (V5H12 H8 checklist + window margins)
- [ ] Prediction check: layer-2 window peak ~= 0.1243 +/-20%

## Formal
- [ ] Formal A full matrix PASS (or exact-prefix STOP, then STOP branch)
- [ ] Formal B + 9-file byte parity + fresh audit
