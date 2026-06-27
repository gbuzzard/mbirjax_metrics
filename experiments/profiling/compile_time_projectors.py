"""
experiments/profiling/compile_time_projectors.py
──────────────────────────────────────────────────
Experiment 3: attribute the COMPILE TIME of the projector ops, split into the three
JAX phases, because compile time is itself nontrivial today and the projectors.py
batching machinery (sum_/concatenate_function_in_batches — the scan/map/vmap nest) is a
refactor candidate partly BECAUSE it is expensive to compile/trace.

For each op we report (all Mac-reachable; the GPU numbers come later from the same script):

  * trace   ms — Python building the jaxpr           (jax.jit(fn).trace(*args))
  * lower   ms — jaxpr -> HLO                         (traced.lower())
  * compile ms — XLA HLO -> executable                (lowered.compile())
  * cold    ms — the REAL first-call a user pays      (trace+lower+compile+exec, production path)
  * warm    ms — steady-state execution               (so we see the compile : run ratio)
  * jaxpr eqns / HLO lines — program-complexity proxies the refactor would shrink

Ops profiled (cone): forward, back-via-PIXEL-kernel driver, back-via-BAND-kernel driver.
The pixel vs band split matters: band's HLO is ~2x bigger (exp 2), so it likely compiles
slower — a cost to weigh against its CPU run-time win.

Run (repo root, `mbirjax` conda env):  python experiments/profiling/compile_time_projectors.py
Writes a YAML table to results/compile_time_<plat>.yaml for the Samba/off-box workflow.
"""
import os
import sys
import time

# Config lives in profiling.env (see profiling_config.py); importing it sets MBIRJAX_NUM_CPU_DEVICES.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from profiling_config import GEOMETRY, sizes_for, N_DEVICES, COMPILE_TRIALS as WARM_TRIALS  # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "tooling", "scaling_tests")))

import mbirjax            # noqa: E402,F401  device-setup-first
import jax                # noqa: E402
import jax.numpy as jnp   # noqa: E402
import performance_tracking as pt   # noqa: E402
import scaling_common as sc          # noqa: E402  (YAML writer)


def _eqn_count(jaxpr):
    """Total primitive equations including those nested inside scan/map/vmap/pjit — the real
    measure of how much program the batching machinery generates (a flat top-level count hides
    the work buried in the scan/map bodies)."""
    n = 0
    eqns = getattr(jaxpr, "eqns", None) or getattr(getattr(jaxpr, "jaxpr", None), "eqns", [])
    for e in eqns:
        n += 1
        for sub in getattr(e, "params", {}).values():
            for inner in (sub if isinstance(sub, (list, tuple)) else [sub]):
                jx = getattr(inner, "jaxpr", inner)
                if hasattr(jx, "eqns"):
                    n += _eqn_count(jx)
    return n


def phase_times(fn, args):
    """Time trace -> lower -> compile separately on a FRESH jit (so each phase actually runs),
    plus jaxpr-eqn and HLO-line complexity counts.  Defensive about the .trace() API."""
    jf = jax.jit(fn)
    t0 = time.perf_counter()
    try:
        traced = jf.trace(*args)
        t_trace = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        lowered = traced.lower()
        t_lower = (time.perf_counter() - t0) * 1e3
        eqns = _eqn_count(getattr(traced, "jaxpr", None))
    except Exception:   # noqa: BLE001 — older API: lower() does trace+lower together
        lowered = jf.lower(*args)
        t_trace, t_lower = float("nan"), (time.perf_counter() - t0) * 1e3
        eqns = float("nan")
    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_compile = (time.perf_counter() - t0) * 1e3
    hlo_lines = len(compiled.as_text().splitlines())
    return dict(trace_ms=t_trace, lower_ms=t_lower, compile_ms=t_compile,
                eqns=eqns, hlo_lines=hlo_lines)


def cold_warm(fn, args):
    """Production cold first-call (trace+lower+compile+exec on the real path) and warm min."""
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    cold_ms = (time.perf_counter() - t0) * 1e3
    warm = []
    for _ in range(WARM_TRIALS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        warm.append((time.perf_counter() - t0) * 1e3)
    return cold_ms, min(warm)


def main():
    plat = jax.devices()[0].platform
    SIZES = sizes_for(plat)                     # per-platform size sweep (CPU small, GPU large)
    jax.block_until_ready(jnp.ones(8).sum())   # init the backend BEFORE the loop (keep it out of size-1 cold)
    print("=" * 100)
    print(f"  COMPILE-TIME attribution | {GEOMETRY} | n={N_DEVICES} {plat} | jax {jax.__version__}")
    print("=" * 100)
    print(f"  {'size':>12} {'op':>12} | {'trace':>8} {'lower':>8} {'compile':>9} | "
          f"{'cold':>9} {'warm':>9} | {'eqns':>6} {'hlo':>6}  (ms)")
    print("  " + "-" * 96)

    rows = []
    config = pt.Config()
    for size in SIZES:
        size_label = "x".join(str(s) for s in size)
        model = pt.make_model(config, GEOMETRY, size)
        if hasattr(model, "configure_devices"):
            model.configure_devices(jax.devices()[:N_DEVICES])
        recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
        idx = jax.device_put(pt.make_indices(model), jax.devices()[0])
        num_pixels, num_slices = len(idx), recon_shape[2]
        sino = pt.to_device(model, pt.make_sinogram(config, size), "sino")
        cyl = pt.to_device(model, pt.make_cylinders(num_pixels, num_slices, config.input_seed), "recon")
        pf = model.projector_functions

        ops = {
            "forward": (lambda c, p: pf.sparse_forward_project(c, p), (cyl, idx)),
            "back_pixel": (lambda s, p: pf.sparse_back_project(s, p), (sino, idx)),
            "back_band": (lambda s, p: pf.sparse_back_project_band(s, p, 0, num_slices), (sino, idx)),
        }
        for op_name, (fn, args) in ops.items():
            ph = phase_times(fn, args)
            cold, warm = cold_warm(fn, args)
            row = dict(size=size_label, op=op_name, cold_ms=cold, warm_ms=warm, **ph)
            rows.append(row)
            print(f"  {size_label:>12} {op_name:>12} | {ph['trace_ms']:>8.0f} {ph['lower_ms']:>8.0f} "
                  f"{ph['compile_ms']:>9.0f} | {cold:>9.0f} {warm:>9.1f} | "
                  f"{ph['eqns']:>6} {ph['hlo_lines']:>6}")

    out = os.path.join(sc.RESULTS_DIR, f"compile_time_{plat}.yaml")
    sc.save_yaml(out, {"platform": plat, "geometry": GEOMETRY, "jax": jax.__version__, "rows": rows})


if __name__ == "__main__":
    main()
