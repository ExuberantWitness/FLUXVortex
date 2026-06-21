"""Decisive G3 go/no-go (careful): is a Warp Tape.backward() CUDA-graph-capturable + correctly
replayable? Test several protocols — the first naive attempt gave rel=1.0, which may be MY misuse
(seed handling / grad reset) rather than a Warp limitation."""
import numpy as np
import warp as wp
from fluxvortex.warp_fsi import config as cfg
from fluxvortex.warp_fsi.config import DTYPE

dev = cfg.DEVICE


@wp.kernel
def _f(x: wp.array(dtype=DTYPE), y: wp.array(dtype=DTYPE)):
    i = wp.tid()
    y[i] = x[i] * x[i] * DTYPE(3.0) + wp.sin(x[i])


def build():
    n = 4096
    xv = np.random.standard_normal(n).astype(cfg.NP_DTYPE)
    x = wp.array(xv, dtype=DTYPE, device=dev, requires_grad=True)
    y = wp.zeros(n, dtype=DTYPE, device=dev, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(_f, dim=n, inputs=[x], outputs=[y], device=dev)
    seed = np.ones(n).astype(cfg.NP_DTYPE)
    ref = 6.0 * xv + np.cos(xv)            # analytic dL/dx for L=sum(y), y=3x²+sin x
    return n, x, y, tape, seed, ref


def main():
    wp.init()
    print("warp", wp.config.version)

    # --- reference (no graph): analytic check first ---
    n, x, y, tape, seed, ref = build()
    y.grad = wp.array(seed, dtype=DTYPE, device=dev)
    tape.backward()
    g_ref = x.grad.numpy().copy()
    print(f"plain backward vs analytic: rel={np.max(np.abs(g_ref-ref))/np.max(np.abs(ref)):.1e}")

    # --- Protocol A: re-seed y.grad + zero x.grad BEFORE each replay (seed outside graph) ---
    n, x, y, tape, seed, ref = build()
    y.grad = wp.array(seed, dtype=DTYPE, device=dev)
    tape.backward()                                  # warmup (compile) + populates
    try:
        with wp.ScopedCapture(device=dev) as cap:
            tape.backward()
        graph = cap.graph
        # replay: reset input grad, re-apply seed, launch
        x.grad.zero_(); y.grad.assign(seed)
        wp.capture_launch(graph); wp.synchronize()
        gA = x.grad.numpy().copy()
        relA = np.max(np.abs(gA - ref)) / np.max(np.abs(ref))
        print(f"Protocol A (reset x.grad + re-seed y.grad before replay): rel={relA:.1e}")
    except Exception as e:
        print("Protocol A FAILED:", repr(e)[:200])

    # --- Protocol B: tape.zero() before the captured backward (zero ALL grads inside graph) ---
    n, x, y, tape, seed, ref = build()
    y.grad = wp.array(seed, dtype=DTYPE, device=dev)
    tape.backward()
    try:
        with wp.ScopedCapture(device=dev) as cap:
            tape.zero()
            y.grad.assign(seed)
            tape.backward()
        graph = cap.graph
        x.grad.zero_()
        wp.capture_launch(graph); wp.synchronize()
        gB = x.grad.numpy().copy()
        relB = np.max(np.abs(gB - ref)) / np.max(np.abs(ref))
        print(f"Protocol B (tape.zero()+re-seed INSIDE graph): rel={relB:.1e}")
    except Exception as e:
        print("Protocol B FAILED:", repr(e)[:200])

    # --- Protocol C: legacy capture_begin/capture_end API ---
    n, x, y, tape, seed, ref = build()
    y.grad = wp.array(seed, dtype=DTYPE, device=dev)
    tape.backward()
    try:
        wp.capture_begin(device=dev)
        try:
            tape.backward()
        finally:
            graph = wp.capture_end(device=dev)
        x.grad.zero_(); y.grad.assign(seed)
        wp.capture_launch(graph); wp.synchronize()
        gC = x.grad.numpy().copy()
        relC = np.max(np.abs(gC - ref)) / np.max(np.abs(ref))
        print(f"Protocol C (capture_begin/end + reset before replay): rel={relC:.1e}")
    except Exception as e:
        print("Protocol C FAILED:", repr(e)[:200])


if __name__ == "__main__":
    main()
