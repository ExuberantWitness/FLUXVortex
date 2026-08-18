"""Baik 2012 W1-W4 runner: bare-core Pterra prescribed wake (best known accuracy).

Uses the generic baik2012.py kinematics for all four cases.
Scoring against the digitized GT (400 phase points per case).
"""
import csv, importlib.util, importlib.metadata as imd, json, sys, time
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import numpy as np

repo = Path("/tmp/fluxv-v5-nextgen")
CASE_ID = sys.argv[1] if len(sys.argv) > 1 else "W2"

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod

executor = load("baik_ex", str(repo / "platform/forward_flight_benchmarks/fluxv_v5h15_baik_w2_executor.py"))
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
print(f"Runtime loaded for {CASE_ID}", flush=True)

# Patch W2_CASE with the desired case's parameters, then use build_w2_movement
from dataclasses import replace as _replace
case = baik.BAIK_2012_CASES[CASE_ID]
executor.W2_CASE = _replace(
    executor.W2_CASE,
    strouhal=case.strouhal,
    reduced_frequency=case.reduced_frequency,
    heave_to_chord=case.heave_to_chord,
    period_s=case.period_s,
)
movement = executor.build_w2_movement(runtime.pterasoftware)
problem = runtime.pterasoftware.problems.UnsteadyProblem(movement=movement, only_final_results=False)

# Compute derived quantities
U = case.reduced_frequency and (np.pi * (1.0 / case.period_s) * case.chord_m / case.reduced_frequency)
qS = 0.5 * 998.2 * U**2 * case.chord_m * case.span_m
steps_per_cycle = 32
n_cycles = 2
period = case.period_s
dt = period / steps_per_cycle
n_steps = steps_per_cycle * n_cycles

# Use same solver as hybrid_w2 (V5H15 coupling, ctrl0 scratch)
coupling = sys.modules["forward_flight_benchmarks.fluxv_v5h15_baik_coupling"]
config_dt = case.period_s / 32
config = runtime.coupling.V5H15BaikCouplingConfig(
    transport_substeps=32, formal_matrix=True, test_mode=False,
    particle_cap=executor.ROW_PARTICLE_CAP, birth_window_refinement=True,
    delta_time_s=config_dt)
level = executor.FORMAL_LEVELS[0]
if CASE_ID == "W2":
    events = executor.build_w2_source_events(runtime)
else:
    # Non-W2 cases: dummy events (committer never fires in ctrl0 coupling)
    events = tuple(tuple(None for _ in range(8)) for _ in range(6))
try:
    committer = executor.BaikW2RowCommitter(runtime, events, level=level)
except Exception:
    class _DummyCommitter:
        def __init__(self): self.captures = []; self.solver = None
        def bind_solver(self, s): self.solver = s
    committer = _DummyCommitter()
solver = coupling.V5H15NativeBaikCouplingSolver(
    problem, row_committer=committer, config=config,
    max_particles=executor.ROW_PARTICLE_CAP, stretch=False, free_wake=False)
committer.bind_solver(solver)
t0 = time.perf_counter()
print(f"Solving {CASE_ID} ({n_steps} steps, T={period}s)...", flush=True)
solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
t1 = time.perf_counter()
print(f"Solve complete in {t1-t0:.1f}s", flush=True)

# Harvest loads (corrected sign convention)
FX, FZ = [], []
for sp in solver.steady_problems:
    f = sp.airplanes[0].forces_W
    if f is not None:
        FX.append(float(f[0])); FZ.append(float(f[2]))
FX, FZ = np.array(FX), np.array(FZ)
steps = len(FX)
t = np.arange(steps) * dt

CL = -FZ / qS
CD = -FX / qS

def lowpass(x, cutoff=1.0):
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), t[1] - t[0])
    X[fr > cutoff] = 0.0
    return np.fft.irfft(X, n=len(x))

CLf, CDf = lowpass(CL), lowpass(CD)
last = slice(steps - steps_per_cycle, steps)
phase_sim = (t[last] - t[last][0]) / period

# Score against GT
gt = {"CL": [], "CD": []}
with open(repo / "docs/forward_flight_large_pitch/reproductions/baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible/scored_phase_samples.csv") as f:
    for row in csv.DictReader(f):
        if row["case_id"] == CASE_ID:
            gt[row["quantity"]].append((float(row["phase"]), float(row["experiment"])))

results = {"case": CASE_ID, "solve_time_s": t1 - t0, "steps": steps}
for q, sim in (("CL", CLf[last]), ("CD", CDf[last])):
    pts = sorted(gt[q])
    gp = np.array([p for p, _ in pts])
    gv = np.array([v for _, v in pts])
    pred = np.interp(gp, phase_sim, sim)
    err = pred - gv
    results[q] = dict(
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        bias=float(np.mean(err)),
        corr=float(np.corrcoef(pred, gv)[0, 1]))
    print(f"{CASE_ID} {q}: RMSE={results[q]['rmse']:.3f} MAE={results[q]['mae']:.3f} "
          f"bias={results[q]['bias']:+.3f} corr={results[q]['corr']:+.3f}", flush=True)

np.savez(f"/tmp/v5h15-paper/baik_{CASE_ID}_curves.npz",
         t=t, CL=CL, CD=CD, CL_f=CLf, CD_f=CDf, phase_sim=phase_sim)
print(f"BAIK {CASE_ID} DONE", flush=True)
