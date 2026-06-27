"""
experiments/profiling/trace_back_projection.py
───────────────────────────────────────────────
Experiment 1 of the fine-grained projector-profiling investigation: produce a
WARM `jax.profiler` trace of ONE projector op (cone back projection) so we can
see where time goes ACROSS the four layers the regression harness fuses into a
single wall-clock number:

  1. host orchestration  — the Python thread pool / band loop / device_put in
                           TomographyModel._back_project_all_bands (and friends)
  2. cross-device comms  — sum_band_to_owner (reduce-scatter), assemble_sharded
  3. compiled XLA program— the jitted scan/map/vmap nest in projectors.py
  4. innermost kernel    — the per-view back kernel's scatter-add

This script does NOT change the harness or the library.  It REUSES the engine's
own input builders (performance_tracking.make_model / make_sinogram /
make_indices / to_device / run_back) so we measure the library exactly the way
the nightly does — just with a profiler wrapped around the WARM (post-compile)
calls instead of a bare timer.

Targets (Mac): CPU only.  At CPU n=1 the back op takes the BANDED sharded driver
(the GPU-only n=1 short-circuit does not fire), so the trace shows the real
sharded machinery with a single device/thread.  Run with MBIRJAX_NUM_CPU_DEVICES
set (see N_DEVICES below) to see the multi-thread fan-out + reduce-scatter.

Run (from the repo root, in the `mbirjax` conda env):

    python experiments/profiling/trace_back_projection.py

Output: a Perfetto trace under experiments/profiling/traces/<tag>/ plus a printed
top-events table.  Open the .json.gz at https://ui.perfetto.dev to explore the
timeline visually.

Reproducibility: all run parameters are clearly-labeled constants at the top
(no command-line args), so a run is fully described by this file.
"""
import os
import sys
import glob
import time
from datetime import datetime

# Config now lives in profiling.env (see profiling_config.py).  Importing it also sets
# MBIRJAX_NUM_CPU_DEVICES (device-setup-first), so it must precede `import mbirjax`.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from profiling_config import GEOMETRY, SIZE, N_DEVICES_LIST, WARMUP, TRACE_ITERS, TOP_N  # noqa: E402

_SCALING = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "tooling", "scaling_tests"))
sys.path.insert(0, _SCALING)
from trace_utils import summarize_perfetto   # noqa: E402

import mbirjax            # noqa: E402,F401 — device-setup side effect; must precede `import jax`
import jax                # noqa: E402
import performance_tracking as pt   # noqa: E402  (reuses make_model/make_sinogram/... — the SAME inputs as the nightly)



def run_one(n_devices):
    avail = jax.devices()
    if n_devices > len(avail):
        print(f"\n  [skip n={n_devices}: only {len(avail)} {avail[0].platform} device(s) available]")
        return
    size_label = "x".join(str(s) for s in SIZE)
    devs = avail[:n_devices]
    plat = devs[0].platform
    print("\n" + "=" * 78)
    print(f"  TRACE  back projection | {GEOMETRY} | {size_label} | n={n_devices} {plat}")
    print(f"  jax {jax.__version__}   devices visible: {len(jax.devices())} {plat}")
    print("=" * 78)

    # Build the model + inputs exactly as the engine does (config defaults are fine for cone).
    config = pt.Config()
    model = pt.make_model(config, GEOMETRY, SIZE)
    if hasattr(model, "configure_devices"):
        model.configure_devices(devs)          # pin to exactly these n devices
    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    idx = pt.make_indices(model)
    num_pixels, num_slices = len(idx), recon_shape[2]
    sino_np = pt.make_sinogram(config, SIZE)
    sino_dev = pt.to_device(model, sino_np, "sino")   # pre-place: measure the op, not the host->device scatter
    print(f"  recon_shape={recon_shape}  num_pixels={num_pixels}  num_slices={num_slices}")

    # Which code path will the trace show?  (is_sharded / band length / #bands)
    info = pt.path_info(model, "back", devs, num_pixels, num_slices)
    print(f"  path: {info}")

    run_fn = lambda: model.sparse_back_project(sino_dev, idx)   # == pt.run_back(model, sino_dev, idx)

    # Warm up: the FIRST call(s) compile every band/batch shape; we trace only warm steady state.
    print(f"\n  warmup x{WARMUP} (compiling) ...", flush=True)
    for _ in range(WARMUP):
        jax.block_until_ready(run_fn())

    # Trace TRACE_ITERS warm iterations.  StepTraceAnnotation delineates each iteration on the
    # timeline; create_perfetto_trace writes a Perfetto-loadable .json.gz alongside the xplane.
    out_dir = os.path.join(_HERE, "traces",
                           f"{datetime.now():%Y%m%d_%H%M%S}_{GEOMETRY}_back_{size_label}_n{n_devices}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  tracing x{TRACE_ITERS} -> {out_dir}", flush=True)
    times = []
    with jax.profiler.trace(out_dir, create_perfetto_trace=True):
        for i in range(TRACE_ITERS):
            with jax.profiler.StepTraceAnnotation("back_project", step_num=i):
                t0 = time.perf_counter()
                jax.block_until_ready(run_fn())
                times.append((time.perf_counter() - t0) * 1e3)

    print(f"\n  warm wall time: min={min(times):.1f} ms  mean={sum(times) / len(times):.1f} ms  "
          f"(n={len(times)})")

    # Find and summarize the Perfetto trace.
    cands = glob.glob(os.path.join(out_dir, "**", "*.json.gz"), recursive=True)
    perfetto = next((c for c in cands if "perfetto" in os.path.basename(c).lower()), cands[0] if cands else None)
    if perfetto:
        print(f"\n  perfetto trace: {perfetto}")
        print(f"  open at https://ui.perfetto.dev  (or load the xplane.pb in TensorBoard)")
        summarize_perfetto(perfetto, TRACE_ITERS)
    else:
        xplanes = glob.glob(os.path.join(out_dir, "**", "*.xplane.pb"), recursive=True)
        print("\n  (no perfetto .json.gz found; xplane(s):)")
        for x in xplanes:
            print(f"    {x}")


def main():
    for n in N_DEVICES_LIST:
        run_one(n)


if __name__ == "__main__":
    main()
