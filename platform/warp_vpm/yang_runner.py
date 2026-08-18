"""Yang 2025 ornithopter: 6 AoA, cycle-mean lift/thrust (gf).

Bare-core Pterra prescribed wake. Four-bar flapping kinematics.
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

executor = load("yang_ex", str(repo / "platform/forward_flight_benchmarks/fluxv_v5h15_baik_w2_executor.py"))
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
from forward_flight_benchmarks.ptera_adapter import build_yang2025_movement
from forward_flight_benchmarks.cases import Yang2025RigidCase
print("Runtime loaded for Yang 2025", flush=True)

case = Yang2025RigidCase()
g = 9.81  # m/s²
gf_per_N = 1000.0 / g  # 1 N = 101.94 gf

# GT data
gt = {}
with open(repo / "docs/forward_flight_large_pitch/reproductions/plev2025/source_data/yang2025_fig11_rigid_digitized.csv") as f:
    for row in csv.DictReader(f):
        gt[float(row["aoa_deg"])] = {
            "test_lift": float(row["test_lift_gf"]),
            "test_thrust": float(row["test_thrust_gf"]),
            "proposed_lift": float(row["proposed_lift_gf"]),
            "proposed_thrust": float(row["proposed_thrust_gf"]),
            "uncertainty": float(row["digitization_uncertainty_gf"]),
        }

aoa_list = sorted(gt.keys())
print(f"Running {len(aoa_list)} AoA conditions: {aoa_list}", flush=True)
coupling = sys.modules["forward_flight_benchmarks.fluxv_v5h15_baik_coupling"]

results = []
for aoa in aoa_list:
    t0 = time.perf_counter()
    movement = build_yang2025_movement(aoa, "full")
    if isinstance(movement, tuple):
        movement = movement[0]

    problem = runtime.pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    dt = movement.delta_time

    config = runtime.coupling.V5H15BaikCouplingConfig(
        transport_substeps=32, formal_matrix=True, test_mode=False,
        particle_cap=executor.ROW_PARTICLE_CAP, birth_window_refinement=True,
        delta_time_s=dt)

    class _DC:
        def __init__(self): self.solver = None
        def bind_solver(self, s): self.solver = s
        def __call__(self, request): return None

    solver = coupling.V5H15NativeBaikCouplingSolver(
        problem, row_committer=_DC(), config=config,
        max_particles=50000, stretch=False, free_wake=False)
    solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    t1 = time.perf_counter()

    # Harvest cycle-mean forces from last cycle
    FX, FZ = [], []
    for sp in solver.steady_problems:
        f_val = sp.airplanes[0].forces_W
        if f_val is not None:
            FX.append(float(f_val[0])); FZ.append(float(f_val[2]))
    FX, FZ = np.array(FX), np.array(FZ)
    steps = len(FX)
    # Estimate steps per cycle from movement period
    period = 1.0 / case.flap_frequency_hz if hasattr(case, 'flap_frequency_hz') else 0.4
    spc = max(1, int(round(period / dt)))
    last = FX[-spc:], FZ[-spc:]

    # Convert to gf (gram-force): F(N) / g * 1000
    lift_gf = float(np.mean(last[1])) / g * 1000.0  # FZ/g * 1000
    thrust_gf = float(np.mean(last[0])) / g * 1000.0  # FX/g * 1000

    gt_data = gt[aoa]
    lift_err = abs(lift_gf - gt_data["test_lift"])
    thrust_err = abs(thrust_gf - gt_data["test_thrust"])
    within = lift_err <= gt_data["uncertainty"] and thrust_err <= gt_data["uncertainty"]

    results.append({
        "aoa": aoa,
        "lift_pred_gf": lift_gf, "lift_gt_gf": gt_data["test_lift"],
        "thrust_pred_gf": thrust_gf, "thrust_gt_gf": gt_data["test_thrust"],
        "lift_err_gf": lift_err, "thrust_err_gf": thrust_err,
        "within_uncertainty": within, "solve_time_s": t1 - t0})

    print(f"  AoA={aoa:.0f}°: lift={lift_gf:.1f}gf (GT {gt_data['test_lift']:.1f}) "
          f"thrust={thrust_gf:.1f}gf (GT {gt_data['test_thrust']:.1f}) "
          f"{('✓' if within else '✗')} [{t1-t0:.1f}s]", flush=True)

lift_errs = [r["lift_err_gf"] for r in results]
thrust_errs = [r["thrust_err_gf"] for r in results]
print(f"\n=== Yang 2025 Summary ===")
print(f"Lift MAE: {np.mean(lift_errs):.1f} gf, Thrust MAE: {np.mean(thrust_errs):.1f} gf")
print(f"Within ±0.4gf: lift {sum(e<=0.4 for e in lift_errs)}/6, thrust {sum(e<=0.4 for e in thrust_errs)}/6")

Path("/tmp/v5h15-paper/yang_results.json").write_text(
    json.dumps({"results": results,
                "lift_mae": float(np.mean(lift_errs)),
                "thrust_mae": float(np.mean(thrust_errs))},
               indent=2, default=float))
print("YANG DONE", flush=True)
