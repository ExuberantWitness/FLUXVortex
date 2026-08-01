# S3ai transport-v2.3 implementation / definition audit

Timestamp: 2026-07-28 19:49:57 +08:00

## Decision

- G2 implementation-diff audit: **PASS**, scientific-chain drift blockers = 0.
- G3 no-history definition regression: **PASS** for dynamic `B/U/R/E` formation and the `leggauss` lazy-load fingerprint. The production dependency-manifest artifact is still absent, so production G3 is not yet complete.
- G4 definition controls: **18/18 PASS**, blockers = 0.
- G5/G6 authorization: **LOCKED / NOT ISSUED**.
- Formal 31-history execution: **NOT AUTHORIZED AND NOT RUN**.
- Scientific stage decision: **UNKNOWN**.
- Production activation, force/HP-state/VES claims, V4.1 modification, 118 sweep, and Fig17/18/19 validation: **NOT AUTHORIZED BY THIS ARTIFACT**.

This artifact audits only the transport/evidence boundary. It changes no aerodynamic equation, constant, grid, kinematics, frozen case, aggregator, or claim state.

## Frozen identities

| Artifact | SHA256 |
|---|---|
| `platform/actual_wake_reachable_pressure_obstruction_v23_one_shot.py` | `c88846a6f1503301ead37c0e03190b756534cbe12f3c2a4a3ef2b619f16eb30a` |
| `platform/tests/test_actual_wake_reachable_pressure_obstruction_v23_one_shot.py` | `5e5a258d238d33fd706fe50b07ec5dff4b1a81d70759d845f00c7c51b632831d` |
| frozen v2.2 wrapper | `ddcb2dccfe315c4dfd978cc04f17fdfbdf99dcc8cf8172f6b3c8d9cea76b428c` |
| frozen guard | `d2f05dd9a4951c082ed3949f59d95dec10f9a052885394d24a6621ec1b295b73` |
| 18:55:56 transport preregistration Markdown | `de5996d1f0a1d3d286a3c40199dce12822526fb2d20ab39c90296481a0f90254` |
| 18:55:56 transport preregistration JSON | `9d8db7f66a03d7450fb2d5f33e66dae553f07fa2b5e40906b9792cac034beac7` |

The reserved production dependency manifest, authorization, result, and marker were all absent at the end of this audit. The v2.3 authorization loader remains deliberately fail-closed.

## Target-environment execution evidence

The tests ran under:

```text
sys.executable =
/home/exuber/anaconda3/envs/fluxvortex/bin/python

pytest =
/home/exuber/anaconda3/lib/python3.11/site-packages/pytest/__init__.py
pytest 9.0.3
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTHONDONTWRITEBYTECODE=1
```

Only the pure-Python pytest runner was appended to the target Python 3.12 process. NumPy, PyYAML, interpreter identity, and runtime dependencies remained those of `fluxvortex`.

Results:

```text
v2.3 definition tests: 25 passed in 15.71 s
v2.3 + frozen v2.2 definition tests:
51 passed, 7 subtests passed in 29.68 s
```

Every definition test independently forbids the frozen collector, canonical mesh builder, and material-wake time marcher. Their call counts were zero. No physical history was executed.

## Closed transport defects

1. The dynamic dependency snapshot was separated from stable runtime/input equality. Legal NumPy polynomial lazy growth no longer changes stable runtime identity.
2. Serialized dependency closure is externally anchored to the authorization-bound manifest, rather than trusting a self-consistent embedded replacement.
3. Every loaded native member must carry a matching `/proc/self/maps` device/inode identity; that evidence is serialized and revalidated.
4. Either canonical result or marker name seals the exact formal result/marker pair and the frozen collector. A fake collector can only use a fully isolated test namespace.
5. Local sources, stable runtime/input, full `U`, old-failure quarantine, marker identity, and the monotonic ledger are rechecked after JSON preparation and immediately before `link()`.
6. A failure after creating the owned canonical hard link rolls that link back. A concurrent third-party result is never deleted or overwritten.
7. Bounded-audit verdict and request/response bindings must equal the verified authorization, not merely be nonempty.

## Mandatory-control result

The fresh same-family read-only audit mapped all 18 preregistered research-profile controls to implementation and executed tests:

1. exact `S0 == B`, including plus/minus controls;
2. legal `B -> B union R`, delta and monotonic ledger;
3. unauthorized Python and same-package exact-file rejection;
4. unauthorized native and native-map identity rejection;
5. changed bytes, same bytes/new inode, symlink/alternate origin, and bytecode substitution;
6. registered-member removal cannot erase `E`;
7. missing mandatory `R`;
8. wildcard, prefix, version-only, duplicate, and malformed manifest rejection;
9. unloaded `U` member drift;
10. wrong first-seen phase;
11. retired authorization SHA, token commitment, and authorization ID;
12. seven retired assets exact-hash stable and old result/latest absent;
13. preexisting and concurrent result/marker no-overwrite behavior;
14. post-collector closure violation consumes the marker and cannot retry;
15. a legal lazy delta plus frozen synthetic `PROTOCOL-NO-GO` publishes through the real aggregate, both validators, ledger, marker, pre-link gate, and atomic writer;
16. post-marker stable-runtime, local-source, full-`U`, and bounded-audit-input drift;
17. manifest/start/end/delta/ledger tampering remains rejected after recomputing internal and outer certificates;
18. all definition tests keep formal collector/mesh/march calls at zero.

## G3 definition evidence and remaining lock

The isolated `leggauss(8)` subprocess dynamically formed a real per-file `B/U/R` manifest from `_loaded_dependency_state()` before and after the call. It did not hard-code the historical counts 170, 179, or 9. It proved:

```text
stable_runtime_before == stable_runtime_after
dependency_delta == actually_added_paths
delta subset U
R subset E
removed_paths == empty
formal collector / mesh / march calls == 0
```

This is a definition/no-history closure proof, not the production closure artifact. Before G5, a production manifest must still be generated in a fresh isolated source-only process, audited per file, and bound to the final wrapper/runtime/source identities.

## Maximum permitted claim

The transport-v2.3 implementation is scientifically non-mutating and its 18 definition controls pass. It is ready to enter production G3 closure formation and a new independent G5 review.

It does **not** establish the 31-history physical result, any pressure obstruction, VES necessity, model improvement, or Fig17/18/19 accuracy.
