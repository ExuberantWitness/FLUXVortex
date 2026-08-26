"""Spike: warp-native CUDA graph capture of surface.evaluate (fixed shapes,
no control flow) with torch FP64 ops inside, per the Warp 1.14 recipe.
Gate: bit-identical replay vs eager, then speedup measurement."""
import sys, time
sys.path.insert(0, "src"); sys.path.insert(0, "platform"); sys.path.insert(0, "platform/warp_vpm")
import numpy as np, torch, warp as wp
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi import kernels_q16_transfer as kxfer
from fluxvortex.warp_fsi.q16_flux_v5m_native import Q16NativeV5MSurface
from forward_flight_benchmarks.rojratsirikul2011_q16 import (
    FORMAL_AERO_GRID, FORMAL_Q16_GRID, ROJ11_A16, make_rojratsirikul2011_q16_model)

mesh, _, _, _ = make_rojratsirikul2011_q16_model(
    chordwise_element_count=FORMAL_Q16_GRID[0], spanwise_element_count=FORMAL_Q16_GRID[1], case=ROJ11_A16)
surface = Q16NativeV5MSurface(mesh, q16_chordwise_elements=FORMAL_Q16_GRID[0],
    q16_spanwise_elements=FORMAL_Q16_GRID[1], aerodynamic_chordwise_panels=FORMAL_AERO_GRID[0],
    aerodynamic_spanwise_panels=FORMAL_AERO_GRID[1], device=config.DEVICE)
state = wp.array(np.ascontiguousarray(mesh.reference_state[None, :]), dtype=config.DTYPE, device=config.DEVICE)
velocity = wp.zeros_like(state)

# eager reference
geom = surface.evaluate(state, velocity)
def grab(g, k):
    v = getattr(g, k)
    return wp.to_torch(v).clone() if not torch.is_tensor(v) else v.clone()
ref = {k: grab(geom, k) for k in ("rings", "normals", "areas", "collocation")}

# Warp-native capture: torch ops run on the warp-derived stream; RELAXED mode
wp.set_stream(wp.Stream("cuda:0"))
torch_stream = wp.stream_to_torch("cuda:0")
with wp.ScopedDevice("cuda:0"):
    with torch.cuda.stream(torch_stream):
        kxfer._CAPTURING = True
    try:
        with wp.ScopedCapture(capture_mode=wp.CaptureMode.RELAXED) as capture:
            geom_static = surface.evaluate(state, velocity)
    finally:
        kxfer._CAPTURING = False
print("capture: OK, graph:", capture.graph)
got = {k: grab(geom_static, k) for k in ref}
same = all(torch.equal(got[k], ref[k]) for k in ref)
print("bit-identical vs eager:", same)
if not same:
    for k in ref:
        print(f"  {k}: max diff {float((got[k]-ref[k]).abs().max()):.3e}")
else:
    wp.capture_launch(capture.graph); wp.synchronize()
    got2 = {k: grab(geom_static, k) for k in ref}
    print("replay still identical:", all(torch.equal(got2[k], ref[k]) for k in ref))
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(200): wp.capture_launch(capture.graph)
    wp.synchronize()
    print(f"graph replay: {(time.perf_counter()-t0)/200*1000:.3f} ms")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(200): surface.evaluate(state, velocity)
    torch.cuda.synchronize()
    print(f"eager: {(time.perf_counter()-t0)/200*1000:.3f} ms")
