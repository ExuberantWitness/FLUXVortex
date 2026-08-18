"""Izraelevitz Fig14 / Scherer 1968: 12 conditions, cycle-averaged CT.

Bare-core Pterra prescribed wake. Each condition ~1s (sinusoidal kinematics).
"""
import csv, importlib.util, importlib.metadata as imd, json, sys, time
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import numpy as np

repo = Path("/tmp/fluxv-v5-nextgen")

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod

executor = load("iz_ex", str(repo / "platform/forward_flight_benchmarks/fluxv_v5h15_baik_w2_executor.py"))
import pterasoftware
ptera_root = Path(pterasoftware.__file__).resolve().parent

def walk(root, prefix):
    out = []
    for py in sorted(root.rglob("*.py")):
        if "__pycache__" in py.parts: continue
        rel = py.relative_to(root).with_suffix("")
        if rel.name == "__init__": rel = rel.parent
        out.append((prefix + "." + ".".join(rel.parts) if rel.parts else prefix, py.resolve()))
    return out

names = [("fluxvortex", (repo/"src/fluxvortex/__init__.py").resolve())]
names += walk(repo/"src/fluxvortex", "fluxvortex")
names += [("forward_flight_benchmarks", (repo/"platform/forward_flight_benchmarks/__init__.py").resolve())]
names += walk(repo/"platform/forward_flight_benchmarks", "forward_flight_benchmarks")
names += walk(ptera_root, "pterasoftware")
names += [("ldvm_fourier", (repo/"platform/ldvm_fourier.py").resolve()),
          ("flap_ldvm", (repo/"platform/flap_ldvm.py").resolve())]

seen = {}
for n, p in names: seen.setdefault(n, p)
SCRATCH = Path("/tmp/v5h15-paper/flux_baik_any_coupling.py").resolve()
seen["forward_flight_benchmarks.fluxv_v5h15_baik_coupling"] = SCRATCH

lf = {leaf: seen.get(executor.MODULE_LEAVES[leaf]) for leaf in executor.MODULE_LEAVES}
lf["v5h15_baik_coupling"] = SCRATCH
for leaf in executor.DISTRIBUTION_LEAVES:
    dist = {"pterasoftware_distribution_metadata":"pterasoftware",
            "fluxvortex_distribution_metadata":"fluxvortex",
            "numpy_distribution_metadata":"numpy",
            "scipy_distribution_metadata":"scipy"}[leaf]
    lf[leaf] = Path(imd.distribution(dist)._path) / "METADATA"

lp, lh = {}, {}
for leaf, path in lf.items():
    lp[leaf] = str(path); lh[leaf] = sha256(Path(path).read_bytes()).hexdigest()
rp, rh = {}, {}
for n, p in seen.items(): rp[n] = str(p); rh[n] = sha256(p.read_bytes()).hexdigest()

class _S: pass
class _T(RuntimeError): pass
api = SimpleNamespace(
    schema_id=executor.EXECUTOR_API_SCHEMA_ID, formal_levels=executor.FORMAL_LEVELS,
    artifact_sink_type=_S, stop_constructor=_T,
    source_event_fields=executor.SOURCE_EVENT_FIELDS, stage_fields=executor.STAGE_FIELDS,
    frontier_node_ids=tuple(f"n{i}" for i in range(9)),
    fixed_probes_gp1_m=((0.05,0.30,0.30),(0.10,0.15,0.40),(0.20,0.45,0.50)),
    dependency_leaf_file_path=lp, dependency_leaf_file_sha256=lh,
    dependency_runtime_module_file_path=rp, dependency_runtime_module_file_sha256=rh)

contract = executor._validated_api(api)
runtime = executor._load_verified_runtime(contract)
from forward_flight_benchmarks.ptera_adapter import build_izraelevitz_scherer_movement
from forward_flight_benchmarks.cases import IzraelevitzSchererCase
print("Runtime loaded for Izraelevitz", flush=True)

case = IzraelevitzSchererCase()
# Compute derived quantities
f = 5.0  # Hz (from J'=U/(f*c)=6, U=3.048, c=0.1016 → f=U/(J'*c)=5.0)
period = 1.0 / f
U = case.freestream_m_s
c = case.chord_m
b = case.aspect_ratio * c  # span = AR * chord
qS = 0.5 * case.rho_kg_m3 * U**2 * c * b

# Load GT
gt_rows = []
with open(repo / "docs/forward_flight_large_pitch/reproductions/unified_fluxv_upgrade_20260812/source_data/izraelevitz2017_fig14_digitized.csv") as fh:
    for row in csv.DictReader(fh):
        if row["data_role"] == "experimental_observation":
            gt_rows.append(row)

conditions = sorted(set((float(r["theta_max_deg"]), float(r["phase_offset_deg"])) for r in gt_rows))
print(f"Running {len(conditions)} conditions...", flush=True)

results = []
coupling = sys.modules["forward_flight_benchmarks.fluxv_v5h15_baik_coupling"]

for theta_max, psi in conditions:
    t0 = time.perf_counter()
    # Build movement for this condition
    movement = build_izraelevitz_scherer_movement(theta_max, psi, "smoke")
    if isinstance(movement, tuple):
        movement = movement[0]

    problem = runtime.pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)

    # Compute dt from movement
    dt = movement.delta_time
    steps_per_cycle = int(round(period / dt))

    # Create solver
    config = runtime.coupling.V5H15BaikCouplingConfig(
        transport_substeps=32, formal_matrix=True, test_mode=False,
        particle_cap=executor.ROW_PARTICLE_CAP, birth_window_refinement=True,
        delta_time_s=dt)

    class _DC:
        def __init__(self): self.captures = []; self.solver = None
        def bind_solver(self, s): self.solver = s
        def __call__(self, request): return None

    solver = coupling.V5H15NativeBaikCouplingSolver(
        problem, row_committer=_DC(), config=config,
        max_particles=50000, stretch=False, free_wake=False)

    solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    t1 = time.perf_counter()

    # Harvest cycle-averaged CT from last cycle
    FX = []
    for sp in solver.steady_problems:
        f_val = sp.airplanes[0].forces_W
        if f_val is not None:
            FX.append(float(f_val[0]))
    FX = np.array(FX)
    steps = len(FX)
    last_cycle = FX[-steps_per_cycle:]
    # CT = -Fx / qS (negative Fx = thrust, positive CT)
    CT = np.mean(last_cycle) / qS  # forces_W[0] is thrust direction

    # Find matching GT points
    gt_match = [r for r in gt_rows
                if float(r["theta_max_deg"]) == theta_max and float(r["phase_offset_deg"]) == psi]
    gt_ct = [float(r["ct"]) for r in gt_match]
    gt_err = [float(r["ct_error_minus"]) for r in gt_match]

    for j, (gct, gerr) in enumerate(zip(gt_ct, gt_err)):
        err = CT - gct
        results.append({
            "theta_max": theta_max, "psi": psi, "replicate": j + 1,
            "CT_pred": CT, "CT_gt": gct, "CT_err": gerr,
            "abs_err": abs(err), "within_bar": abs(err) <= gerr})

    status = "✓" if any(r["within_bar"] for r in results[-len(gt_match):]) else "✗"
    print(f"  θ={theta_max}° ψ={psi}°: CT={CT:+.4f} vs GT {gt_ct[0]:.4f} "
          f"|Δ|={abs(CT-gt_ct[0]):.4f} {status} [{t1-t0:.1f}s]", flush=True)

# Summary
all_err = [r["abs_err"] for r in results]
within = sum(1 for r in results if r["within_bar"])
print(f"\n=== Izraelevitz Fig14 Summary ===")
print(f"Total markers: {len(results)}, within error bars: {within}/{len(results)} ({100*within/len(results):.0f}%)")
print(f"MAE: {np.mean(all_err):.4f}, RMSE: {np.sqrt(np.mean(np.array(all_err)**2)):.4f}")

# Save
Path("/tmp/v5h15-paper/izra_results.json").write_text(
    json.dumps({"results": results,
                "mae": float(np.mean(all_err)),
                "rmse": float(np.sqrt(np.mean(np.array(all_err)**2))),
                "within_bars": f"{within}/{len(results)}"},
               indent=2, default=float))
print("IZRAELEVITZ DONE", flush=True)
