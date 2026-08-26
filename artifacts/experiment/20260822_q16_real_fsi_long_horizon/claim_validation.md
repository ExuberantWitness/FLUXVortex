# Claim Validation

| Claim | Evidence | Verdict |
|---|---|---|
| one live Q16 owner can advance 8 steps with mandatory LEV, joint TEV and free wake | long-horizon pytest plus exact generation, particle and wake sequences in `metrics.json` | supported for the frozen damped fixture |
| segmented 4+4 execution preserves one exact trajectory lineage | prefix chain is the fourth record parent of the final chain; independent validator passes | supported |
| structural damping is present in both equations and accounting | nonzero damping regression and positive per-step damping work with closure <= `2.46e-12` | supported |
| indefinite tangents are no longer sent through PCG only | steps 6 and 8 record 3 fallbacks and 216 GMRES iterations with accepted original residuals | supported |
| joint LEV active classification closes after coupled solve | step 5 crosses from 36 to 48 particles and the formerly failing G3 coordinate commits | supported for this observed activation |
| implementation is multi-cycle validated | fixture has no frequency and spans no defined cycles | not supported |
| implementation is experimentally accurate | no experimental oracle is used in this run | not supported |
| undamped `E=1e8` fixture is stable for 8 steps | fifth coordinate reaches geometric/nonlinear failure boundary | contradicted |

