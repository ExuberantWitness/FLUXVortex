# Executable Claim Runtime

`claim_runtime` turns `claim_nodes/*.yaml` into a validated runtime DAG.

For every `gpu_run_twist()` call it:

1. selects nodes by closure profile;
2. validates scientific dependencies, runtime data dependencies, exclusivity,
   node state/role and frozen implementation hashes;
3. executes nodes in topological order;
4. books every named force through one `ForceLedger`;
5. refuses to return if the ledger does not reproduce the solver body force;
6. returns the topology, implementation identity, parameter provenance,
   contributions and guards in the result dictionary.

The Warp UVLM kernels and their time integration remain the validated N1
numerical kernel. Historical non-production switches are isolated behind the
`LEGACY` compatibility node.

Public result additions:

- `claim_manifest`
- `claim_contributions`
- `claim_guards`

Existing `gpu_run_twist()` arguments and result fields are unchanged.
