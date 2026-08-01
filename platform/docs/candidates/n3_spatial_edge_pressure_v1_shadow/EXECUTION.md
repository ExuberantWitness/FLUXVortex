# N3 spatial edge-pressure v1 shadow execution contract

Python:

```text
/home/exuber/anaconda3/envs/fluxvortex/bin/python
```

Candidate outputs:

```text
platform/docs/candidates/n3_spatial_edge_pressure_v1_shadow/runs/<timestamp>/
```

Order:

1. unit tests and tiny GPU identity test;
2. `smoke3 --quick`;
3. quadrature 24 sentinels and the G1 quadrature early-stop scorer;
4. only if quadrature passes, run the half-time-step high-twist sentinel;
5. only if all of G1 passes, run `representative32 --quick`;
6. score candidate against its same-call V4.1 counterfactual;
7. stop on G0/G1/G2 failure;
8. otherwise run `confirmed151`.

The original phrase “sentinel total force” was ambiguous.  Its
post-data-exposure interpretation and the already visible diagnostic values
are disclosed in `DATA_EXPOSURE_ADDENDUM.md`; the `0.5%` threshold is
unchanged.  Formal runs use the same candidate id and differ only through the
recorded numerical coordinate:

```text
python platform/lb_sweep_candidate.py \
  --candidate-id n3_spatial_edge_pressure_v1_shadow \
  --closure n3_spatial_edge_pressure_v1_shadow \
  --scope smoke3 --quick

python platform/lb_sweep_candidate.py \
  --candidate-id n3_spatial_edge_pressure_v1_shadow \
  --closure n3_spatial_edge_pressure_v1_shadow \
  --scope smoke3 --quick \
  --model-arg spatial_p2_quadrature=24
```

The runner must not read measured forces during solver execution.  Scoring
and plotting occur only after the candidate result file is closed.

No result from `n3_spatial_pressure_v0` may seed this campaign.
