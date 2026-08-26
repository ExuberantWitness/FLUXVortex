# Q16 shared-node mesh result

Result: **PASS for the registered small shared-mesh gate**.

Two chordwise Q16 elements use 28 unique nodes (168 DOF), sharing exactly one
four-node cubic edge; a 2x2 mesh uses 49 unique nodes. CPU energy/force/Jv and
consistent-mass identities pass. CUDA force, mass and analytic Jv match the
independent CPU mesh oracle, and deterministic per-global-node CSR gathering
is bitwise repeatable. The same shared-node path now accepts the fixed
MITC16+ANS/EAS condensed element. Its CUDA relative L2 errors are
`4.25e-15` (force), `8.76e-16` (Jv), and `1.84e-16` (mass). No
floating-point atomic scatter or CPU numerical fallback is used.

This result does not yet include boundary-condition ownership, nonlinear time
integration, or a production wing mesh.
