"""Baik W1-W4 runner with Roccia augmented-system LEV."""
import csv, importlib.util, importlib.metadata as imd, json, sys, time
from dataclasses import replace as _replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import numpy as np

repo = Path("/tmp/fluxv-v5-nextgen")
sys.path.insert(0, str(repo / "platform"))
sys.path.insert(0, str(repo / "src"))
from warp_vpm.augmented_lev import AugmentedLEVSolver

LESP_TRIGGER = 0.003  # circulation threshold for LEV activation

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod

executor = load("aex", str(repo / "platform/forward_flight_benchmarks/fluxv_v5h15_baik_w2_executor.py"))
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
from forward_flight_benchmarks import baik2012 as baik
coupling = sys.modules["forward_flight_benchmarks.fluxv_v5h15_baik_coupling"]
print("Runtime loaded for Augmented LEV", flush=True)


class BaikAugmentedSolver(AugmentedLEVSolver, coupling.V5H15NativeBaikCouplingSolver):
    def __init__(self, problem, U, c, period, **kw):
        super().__init__(problem, **kw)
        self._init_augmented_lev(U, c, period)
        self._lesp_trigger = LESP_TRIGGER


def run_case(case_id):
    case = baik.BAIK_2012_CASES[case_id]
    executor.W2_CASE = _replace(executor.W2_CASE,
        strouhal=case.strouhal, reduced_frequency=case.reduced_frequency,
        heave_to_chord=case.heave_to_chord, period_s=case.period_s)
    movement = executor.build_w2_movement(runtime.pterasoftware)
    dt = case.period_s / 32
    U = np.pi * (1.0 / case.period_s) * case.chord_m / case.reduced_frequency

    problem = runtime.pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)
    config = runtime.coupling.V5H15BaikCouplingConfig(
        transport_substeps=32, formal_matrix=True, test_mode=False,
        particle_cap=executor.ROW_PARTICLE_CAP, birth_window_refinement=True,
        delta_time_s=dt)

    class _DC:
        def __init__(self): self.solver = None
        def bind_solver(self, s): self.solver = s
        def __call__(self, request): return None

    solver = BaikAugmentedSolver(
        problem, U=U, c=case.chord_m, period=case.period_s,
        row_committer=_DC(), config=config,
        max_particles=50000, stretch=False, free_wake=False)

    t0 = time.perf_counter()
    solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
    t1 = time.perf_counter()

    qS = 0.5 * 998.2 * U**2 * case.chord_m * case.span_m
    FX, FZ = [], []
    for sp in solver.steady_problems:
        f = sp.airplanes[0].forces_W
        if f is not None:
            FX.append(float(f[0])); FZ.append(float(f[2]))
    FX, FZ = np.array(FX), np.array(FZ)
    steps = len(FX)
    t = np.arange(steps) * dt
    CL = -FZ / qS; CD = -FX / qS

    def lp(x):
        X = np.fft.rfft(x); fr = np.fft.rfftfreq(len(x), dt)
        X[fr > 1.0] = 0; return np.fft.irfft(X, n=len(x))
    CLf, CDf = lp(CL), lp(CD)
    last = slice(steps - 32, steps)
    phase = (t[last] - t[last][0]) / case.period_s

    gt = {"CL": [], "CD": []}
    with open(repo / "docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible/scored_phase_samples.csv") as f:
        for row in csv.DictReader(f):
            if row["case_id"] == case_id:
                gt[row["quantity"]].append((float(row["phase"]), float(row["experiment"])))

    r = {"case": case_id, "time": t1 - t0, "particles": solver.pf.n}
    for q, sim in (("CL", CLf[last]), ("CD", CDf[last])):
        pts = sorted(gt[q])
        gp = np.array([p for p, _ in pts]); gv = np.array([v for _, v in pts])
        pred = np.interp(gp, phase, sim)
        err = pred - gv
        r[q] = dict(rmse=float(np.sqrt(np.mean(err**2))), bias=float(np.mean(err)),
                    corr=float(np.corrcoef(pred, gv)[0, 1]))
        print(f"  {case_id} {q}: RMSE={r[q]['rmse']:.3f} bias={r[q]['bias']:+.3f} corr={r[q]['corr']:+.3f}", flush=True)
    return r


if __name__ == "__main__":
    results = []
    for case in ["W1", "W2", "W3", "W4"]:
        print(f"\n=== {case} ===")
        results.append(run_case(case))

    cl = [r["CL"]["rmse"] for r in results]
    cd = [r["CD"]["rmse"] for r in results]
    print(f"\n=== Augmented LEV Summary ===")
    print(f"CL macro RMSE: {np.mean(cl):.3f} (bare: 0.793, SOTA: 0.594)")
    print(f"CD macro RMSE: {np.mean(cd):.3f} (bare: 0.420, SOTA: 0.346)")
    Path("/tmp/v5h15-paper/augmented_lev_results.json").write_text(
        json.dumps(results, indent=2, default=float))
    print("DONE", flush=True)
