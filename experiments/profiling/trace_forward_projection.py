"""
experiments/profiling/trace_forward_projection.py
───────────────────────────────────────────────────
Trace cone FORWARD projection — the projector flagged as the dominant GPU cost (exp 3: warm 599 ms
@256³ ≈ 8.7× the cone back pixel-kernel) but never broken down by kernel.  Mirrors
trace_back_projection.py: warm `jax.profiler.trace` of `model.sparse_forward_project`, self-time +
per-track summary (shared `trace_utils.summarize_perfetto`).  Reuses the engine's input builders.

Hypothesis to test (key_findings.md): the cone forward-kernel
`ConeBeamModel.forward_project_pixel_batch_to_one_view` (mbirjax/cone_beam.py:275) rolls a
`jax.lax.map` over detector rows (:470) — does that materialize/serialize the way the back
pixel-kernel's `lax.map`+transpose does?  The fusion-family ranking should show where the time goes.

Run (repo root, the active env):  python experiments/profiling/trace_forward_projection.py
"""
import os
import sys
import glob
import time
from datetime import datetime

# Config lives in profiling.env (see profiling_config.py); importing it sets MBIRJAX_NUM_CPU_DEVICES
# (device-setup-first), so it must precede `import mbirjax`.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from profiling_config import GEOMETRY, size_for, N_DEVICES_LIST, WARMUP, TRACE_ITERS, TOP_N  # noqa: E402

_SCALING = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "tooling", "scaling_tests"))
sys.path.insert(0, _SCALING)
from trace_utils import summarize_perfetto   # noqa: E402

import mbirjax            # noqa: E402,F401 — device-setup-first
import jax                # noqa: E402
import performance_tracking as pt   # noqa: E402


def run_one(n_devices):
    avail = jax.devices()
    if n_devices > len(avail):
        print(f"\n  [skip n={n_devices}: only {len(avail)} {avail[0].platform} device(s) available]")
        return
    devs = avail[:n_devices]
    plat = devs[0].platform
    SIZE = size_for(plat)                       # per-platform size (CPU small, GPU large)
    size_label = "x".join(str(s) for s in SIZE)
    print("\n" + "=" * 78)
    print(f"  TRACE  forward projection | {GEOMETRY} | {size_label} | n={n_devices} {plat}")
    print(f"  jax {jax.__version__}   devices visible: {len(jax.devices())} {plat}")
    print("=" * 78)

    config = pt.Config()
    model = pt.make_model(config, GEOMETRY, SIZE)
    if hasattr(model, "configure_devices"):
        model.configure_devices(devs)
    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    idx = pt.make_indices(model)
    num_pixels, num_slices = len(idx), recon_shape[2]
    # Forward's input is the recon (voxel cylinders); pre-place on the recon (slice-sharded) device form
    # OUTSIDE the timing loop — measure the op, not the host->device scatter.
    cylinders = pt.make_cylinders(num_pixels, num_slices, config.input_seed)
    cyl_dev = pt.to_device(model, cylinders, "recon")
    print(f"  recon_shape={recon_shape}  num_pixels={num_pixels}  num_slices={num_slices}")
    print(f"  path: {pt.path_info(model, 'forward', devs, num_pixels, num_slices)}")

    run_fn = lambda: model.sparse_forward_project(cyl_dev, idx)   # == pt.run_forward(model, cyl_dev, idx)

    print(f"\n  warmup x{WARMUP} (compiling) ...", flush=True)
    for _ in range(WARMUP):
        jax.block_until_ready(run_fn())

    out_dir = os.path.join(_HERE, "traces",
                           f"{datetime.now():%Y%m%d_%H%M%S}_{GEOMETRY}_forward_{size_label}_n{n_devices}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  tracing x{TRACE_ITERS} -> {out_dir}", flush=True)
    times = []
    with jax.profiler.trace(out_dir, create_perfetto_trace=True):
        for i in range(TRACE_ITERS):
            with jax.profiler.StepTraceAnnotation("forward_project", step_num=i):
                t0 = time.perf_counter()
                jax.block_until_ready(run_fn())
                times.append((time.perf_counter() - t0) * 1e3)

    print(f"\n  warm wall time: min={min(times):.1f} ms  mean={sum(times) / len(times):.1f} ms  "
          f"(n={len(times)})")

    cands = glob.glob(os.path.join(out_dir, "**", "*.json.gz"), recursive=True)
    perfetto = next((c for c in cands if "perfetto" in os.path.basename(c).lower()), cands[0] if cands else None)
    if perfetto:
        print(f"\n  perfetto trace: {perfetto}")
        print(f"  open at https://ui.perfetto.dev  (or `tensorboard --logdir {out_dir}`)")
        summarize_perfetto(perfetto, TRACE_ITERS, TOP_N)
    else:
        for x in glob.glob(os.path.join(out_dir, "**", "*.xplane.pb"), recursive=True):
            print(f"    xplane: {x}")


def main():
    for n in N_DEVICES_LIST:
        run_one(n)


if __name__ == "__main__":
    main()
