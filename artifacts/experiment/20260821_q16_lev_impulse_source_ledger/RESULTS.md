# Q16 LEV Impulse Source Ledger Results

## Outcome

`PASS / GO` for causal spanwise source ownership of the active-LEV impulse.

`PARTIAL / STOP` for a complete Q16 structural commit. The source ledger now
identifies where each impulse contribution came from, but the corresponding
Q16 virtual-work operator has not yet been implemented and audited.

## Main Results

| Metric | Observed | Gate |
|---|---:|---:|
| CUDA LEV particles | `24` | nonzero |
| source strips | `3` | exact panel topology |
| particles per strip | `8 / 8 / 8` | causal ring-leg ownership |
| global impulse-force norm | `153.23676513670927` | finite, nonzero |
| strip/global closure max error | `3.197442310920451e-13` | within scaled float64 gate |
| pre/post packet SHA | `e7c051e9...e7659` | bitwise unchanged |
| focused tests | `5/5` | all pass, zero warnings |
| joint tests | `146/146` | all pass |

The final two-step strip forces in world coordinates are:

```text
strip 0: [39.335006370811314,  0.42301747888759783, -12.165268014727582]
strip 1: [71.37558197104588,  -1.0408340855860843e-15, -6.7793610594892195]
strip 2: [39.33500637081109,  -0.4230174788875961,  -12.165268014727415]
```

## Mechanism

Every active leg emitted by `add_ring_particles` receives the ring's exact
span index on CUDA as `int64`. The identity stays attached through WRK3
convection and compaction/removal and is included in the incremental solver's
scientific digest.

The solver keeps its original global free-vortex plus bound-sheet impulse
reduction unchanged. In parallel it accumulates the same physical operands by
source strip, differences them against the previous strip impulse, and rejects
if their sum fails to close the original global force. It also retains each
strip's current pair of real leading-edge endpoints in world coordinates.

## Claim Boundary

- Supported: the LEV impulse is no longer an ownerless global scalar/vector;
  every particle contribution has a persistent causal span-strip identity.
- Supported: the additional decomposition does not alter the frozen global
  load packet; its SHA and exact impulse components are unchanged.
- Supported: separated LEV, joint TEV, free wake, CUDA and float64 remain
  active on the tested path.
- Not yet supported: applying the strip impulse to Q16 generalized coordinates.
  A source location alone is not yet a proof of structural work conjugacy.
- No paper data, GT or scorer was accessed; this is an auxiliary integration
  result, not a paper-accuracy result.

## Next Action

Build the exact Q16 impulse-work operator from the two Q16 interpolation rows
that generated each real leading-edge strip endpoint. Apply one half of the
strip force at each endpoint, then independently verify resultant, midpoint
moment and virtual work before allowing an aerodynamic transaction to commit.
