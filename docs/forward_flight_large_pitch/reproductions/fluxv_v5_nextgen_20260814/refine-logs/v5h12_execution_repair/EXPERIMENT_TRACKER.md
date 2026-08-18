# V5H12 execution-only repair tracker

| ID | Gate | Evidence | Required verdict | Status | Notes |
|---|---|---|---|---|---|
| V5H12-G0-001 | frozen-input capture | `FREEZE_INPUTS.json` | all named V5H11 leaves and formal-A hashes recorded | PASS_DOC | no scientific execution |
| V5H12-G0-002 | observation boundary | plan/checklist review | GT/scorer unopened; `observation_access=none` | PASS_DOC | inherited from V5H11 |
| V5H12-G0-003 | immutable namespace amendment | four new target paths plus V5H11 hash controls | implementation may create only V5H12 runner/executor/tests; all V5H11 leaves immutable | PASS_DOC | amended before implementation |
| V5H12-G1-001 | seven-plus-one ownership | `test_actual_coupling_record_carries_exact_seven_observer_fields_and_parses` (real `transport_v5h11_committed_layer` N=1 record) | seven observer fields accepted; normalized invariant read only from compact record | PROVISIONAL PASS_TEST | RED reproducible on demand, but no original UTC/stdout snapshot proves RED-before-GREEN ordering (P0 disclosure) |
| V5H12-G1-002 | strict negative parser matrix | `test_stage_evidence_rejects_duplicate_or_incomplete_json`, `test_extra_eighth_observer_field_is_rejected_and_payload_bytes_are_binding`, `test_record_owned_normalized_invariant_rejects_hostile_values`, `test_record_owned_normalized_invariant_accepts_exact_gate_and_is_bitwise_copied` | every attack fails closed without fabricated evidence | PASS_TEST | hostile values: None/int/NaN/Inf/negative/nextafter-over-gate; equal gate accepted bitwise |
| V5H12-G1-003 | source-before-stage durability | P1 closed: implicit source-append fallback REMOVED from `commit_completed_layer` (hard pre-append requirement + unconditional byte-identity guard); directed tests `test_commit_requires_pre_appended_durable_source_and_leaves_sink_unchanged`, `test_exactly_once_contract_rejects_identical_duplicate_source_append`, `test_exactly_once_contract_rejects_inconsistent_duplicate_source_append`, `test_source_exactly_once_then_stages_then_commit_ends_with_one_source`, `test_source_append_failure_blocks_stages_and_clean_retry_succeeds`, `test_commit_rejects_same_key_source_with_different_canonical_bytes` | source parent durable before first stage and appended exactly once | PASS_TEST | commit-without-source tested from the empty sink because a 96-stage prefix without source is unreachable via the public ABI (each stage append is rejected first); pre-appending the NEXT layer source after a commit is legal and asserted |
| V5H12-G1-004 | exact conversion STOP coordinate | `test_conversion_failure_raises_exact_unbegun_next_stage_stop[0-1]/[1-2]`, `test_conversion_stop_terminal_preserves_unbegun_next_stage_coordinate`, `test_conversion_stop_rejects_hostile_coordinate_with_last_good_stop` | exact next six-coordinate retained; `(32,1,4,3,1,1)`, `stage_began=false` at first conversion | PASS_TEST | runner branch gated on code `stage_evidence_conversion_error` + phase `artifact_stage_conversion` only; generic `(32,None,4,3,None,None)` unchanged; hostile coordinate falls back to last-good publishable STOP |
| V5H12-G1-005 | frozen historical leaves | pre/post SHA comparison (re-run after final Black reformat) | all eight V5H11 source/test controls byte-identical | PASS | no drift; governance/formal-A hashes re-verified at H0 and after implementation |
| V5H12-G1-006 | focused regression | P2 closed for the V5H12 scope: sys.modules isolation fixtures added to the two V5H12 test files (`pristine_audited_modules`/`restore_runtime_modules_after`; autouse `_pristine_audited_runtime_modules`) WITHOUT touching any runner/executor gate; V5H12 executor+runner in ONE process = 84 passed / 0 failed; per-file fresh-process: 33/51/20/42/11/25 all PASS; static gates clean | all pass (incl. §9.3 joint command) | PARTIAL PASS / amendment required | literal §9.3 six-file command still fails 12 tests, ALL in frozen V5H11 files; the frozen V5H11 trio (executor+runner+coupling, no V5H12 files) reproduces the identical 12 failures, proving structural collection-time interference between frozen files (coupling test module-level real imports vs. executor origin-attest and runner runtime-inventory tests). Unfixable inside the 4-file allowed scope; requires explicit amendment (per-file fresh-process joint definition, or authorize a conftest.py, or unfreeze V5H11 tests) |
| V5H12-G1-007 | traceable hostile audit (H6 closure) | `H6_FRESH_HOSTILE_AUDIT_20260816.md` (SHA `4801356299d0270d77a83b05346a128e7b864cc209a20c5f61474f828eea5e9b`): round-1 reviewer agent_1c5b46a9 FAIL (1 must-fix: serialized re-validation omitted stop_code, first-stage conversion STOP crashed publication self-check; 1 should-note) -> fixed + mutation-verified -> round-2 reviewer agent_65ccac08 PASS no must-fix; attack-matrix stdout embedded | no must-fix on final leaves | PASS | fix included directed publish+reverify test `test_first_stage_conversion_stop_publishes_and_reverifies_byte_identically` (mutation check: fails at runner:3791 with fix removed) and `test_commit_rejects_same_key_source_with_different_canonical_bytes` |
| V5H12-G2-001 | dependency re-freeze | fresh manifest+token at `/tmp/fluxv-v5h12-audit-20260816-4W8c03/` (41 leaves + 56 runtime modules, binds final leaves runner `5e0777d8…`/executor `5c74a9ff…`); `_verified_dependency_audit` PASS; `_load_formal_executor` full chain PASS (load+attest+observed capture); old V5H11 token `DependencyFreezeError: dependency audit token schema_id is invalid` (fail closed) | repaired leaves bound; old token rejects | PASS (evidence complete) | disposable smoke activation additionally gated on the §9.3 joint-command amendment (see G1-006); no Ptera case constructed beyond the token-closure loader import |
| V5H12-G2-002 | disposable real smoke | fresh `N32/layer1` process | inherited mechanics/artifact-preflight gates pass; no durable formal rows | BLOCKED_BY_G2_001 | no GT, no selection |
| V5H12-G3-001A | formal A | fresh `N=32,64,128 x layers=1,2,3` | complete inherited matrix PASS or exact-prefix STOP | BLOCKED_BY_G2_002 | new immutable destination |
| V5H12-G3-001B | formal B | independent fresh process | complete inherited matrix PASS or exact-prefix STOP | BLOCKED_BY_G3_001A | new immutable destination |
| V5H12-G3-002 | deterministic semantic artifact | first nine A/B semantic files | byte-identical; provenance differs | BLOCKED_BY_G3_001B | no manual normalization |
| V5H12-G3-003 | fresh artifact audit | read-only A/B audit | dependency, schema, hashes, gates, and prefixes close | BLOCKED_BY_G3_002 | audit failure is STOP |
| V5H12-G4-001 | paper-data unlock | inherited unlock-token contract | remain sealed unless every prior gate passes | BLOCKED_BY_G3_003 | V5H12 does not relax M4 |

Current frontier (updated 2026-08-16 after evidence closure P0-P4): P0 done
(status corrections), P1 done (exactly-once closed, fallback removed, 6
directed tests), P2 done for the V5H12 scope (same-process pair 84 passed / 0
failed via sys.modules isolation fixtures that touch no gate) with the literal
six-file §9.3 command blocked by proven frozen-V5H11-internal collection-time
interference (frozen trio alone reproduces the same 12 failures) — an explicit
amendment is required before "joint command green" can be certified, P3 done
(`H6_FRESH_HOSTILE_AUDIT_20260816.md`, two-round reviewer trail, round-2 PASS
no must-fix, mutation-verified fix for the serialized stop_code omission), P4
done (fresh manifest/token verified, full executor-load closure PASS, old
V5H11 token fail closed). Final leaves: runner
`5e0777d82147827a0ebcd9520f3a6cfdade592bc392d3306ac77e7a9085f05fe`, executor
`5c74a9ffe245a0212aacf06067c477c6ddcc384e1c71b97a2aa1bd017bfb7053` (unchanged
since round 1), runner test
`e8b3de1271cc8cffa09e6e1252a68595662a2b2328bbdc943a18bdb5b3c3b2fa`, executor
test `3990bb32309eb69d858e768a4d476ff42e815524afdf38108aaf29314f875b07`
(unchanged). Next admissible action: the §9.3 joint-command governance
amendment; after it is accepted, the fresh disposable N32/layer1 smoke
(G2-002) may run. Earlier historical notes:
(1) the conversion STOP wrap types every stage-loop exception as conversion per
the prereg wording, and (2) `FORMAL_AUDIT_SCOPE` label change
`v5h11_b3_no_gt_formal_execution` -> `v5h12_execution_repair_no_gt_formal_execution`
was part of the pre-existing mechanical fork intermediate state).  Final leaf
SHA-256 after Black: runner
`063725d1512df574b969f302b12199e4ff82c0ac52c6be15b34043cf640d282b`, executor
`5c74a9ffe245a0212aacf06067c477c6ddcc384e1c71b97a2aa1bd017bfb7053`, runner test
`8612d52ba330fbe9f287fdd57190aa3dcfee6d1d436e96a351c469269395d4a9`, executor
test `b155af9a224dfa7f2b7f32f5a6903ed0dbecdf7b9245a82d6df4b1dce2f9fbd6`.
The next admissible action is `V5H12-G2-001` (fresh dependency manifest and
external audit token; old V5H11 token must reject the changed leaves).
