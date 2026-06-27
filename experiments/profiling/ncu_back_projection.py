"""
experiments/profiling/ncu_back_projection.py
───────────────────────────────────────────────
Experiment 4 (GPU only): a MINIMAL driver so Nsight Compute (`ncu`) can profile the
dominant back-projection kernels and answer the one question the JAX trace + static
analysis can't — **is the kernel at the HBM bandwidth roofline, or is there headroom?**

The exp-1 GPU trace named the two kernels that own ~83% of cone back (n=1, pixel kernel):
    loop_add_fusion                  (~58% — the back-projection accumulate)
    loop_dynamic_update_slice_fusion (~25% — the rolled lax.map scatter-write)

This script just runs that op cleanly (warm up to compile, then a few warm calls); ncu does the
profiling from OUTSIDE via the command line.  Keeping the call count tiny keeps ncu's kernel-replay
time bounded (ncu re-runs each profiled kernel many times to collect counters).

Run it UNDER ncu (on the cluster, GPU env).  Start SMALL — SpeedOfLight only, the two target
kernels, a few launches — then widen if needed:

    mkdir -p experiments/profiling/ncu
    ncu --profile-from-start off --set basic \
        --kernel-name "regex:add_fusion|dynamic_update_slice" \
        --launch-count 6 --target-processes all \
        --csv --log-file experiments/profiling/ncu/back_pixel_256.csv \
        python experiments/profiling/ncu_back_projection.py

Notes / likely iteration (ncu flags are version-sensitive — Nsight 2025.1 here):
  * `--profile-from-start off` + the script's cudaProfilerStart/Stop (cuda_profiler.profiler_range)
    scope profiling to the warm region, skipping JAX import + compile/warmup — the bulk of the old
    ~8 min wall time.  Without it, ncu instruments the whole process (incl. compile autotuning).
  * `--set basic` ≈ SpeedOfLight (compute vs memory throughput %) — the roofline read; `--set full`
    adds occupancy/memory-workload/scheduler (much slower, do later if SoL is ambiguous).
  * `--kernel-name regex:...` targets only the two kernels; drop it (with a small --launch-count) if
    the XLA kernel symbols don't match the fusion names and nothing gets profiled.
  * roofline read: memory-bound  => Memory throughput % ≫ Compute(SM) throughput %  AND DRAM near peak;
    if BOTH are well below 100%, the kernel is latency/occupancy-bound, not at any roofline (headroom).
  * the .csv lands on the Samba mount for off-box reading; a .ncu-rep (`-o ...`) opens in the Nsight UI.
"""
import os
import sys
import time

# Config lives in profiling.env (see profiling_config.py); importing it sets MBIRJAX_NUM_CPU_DEVICES.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from profiling_config import GEOMETRY, size_for, N_DEVICES, WARMUP, PROFILE_CALLS  # noqa: E402
from cuda_profiler import profiler_range   # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "tooling", "scaling_tests")))

import mbirjax            # noqa: E402,F401  device-setup-first
import jax                # noqa: E402
import performance_tracking as pt   # noqa: E402


def main():
    plat = jax.devices()[0].platform
    if plat != "gpu":
        print(f"  WARNING: backend is {plat}, not gpu — ncu profiling is a GPU-only step.")
    devs = jax.devices()[:N_DEVICES]
    SIZE = size_for(plat)                       # per-platform size (CPU small, GPU large)
    size_label = "x".join(str(s) for s in SIZE)
    print(f"  ncu driver: {GEOMETRY} back | {size_label} | n={N_DEVICES} {plat}")

    config = pt.Config()
    model = pt.make_model(config, GEOMETRY, SIZE)
    if hasattr(model, "configure_devices"):
        model.configure_devices(devs)
    idx = pt.make_indices(model)
    sino = pt.to_device(model, pt.make_sinogram(config, SIZE), "sino")
    run_fn = lambda: model.sparse_back_project(sino, idx)

    for _ in range(WARMUP):                 # compile OUTSIDE the profiled region (ncu --profile-from-start off skips it)
        jax.block_until_ready(run_fn())
    t0 = time.perf_counter()
    with profiler_range():                  # ncu profiles ONLY this region (cudaProfilerStart/Stop)
        for _ in range(PROFILE_CALLS):
            r = run_fn()
        jax.block_until_ready(r)            # finish the profiled kernels before cudaProfilerStop
    dt = (time.perf_counter() - t0) / PROFILE_CALLS * 1e3
    print(f"  warm time ~{dt:.1f} ms/call  (profiled region: {PROFILE_CALLS} call(s))")


if __name__ == "__main__":
    main()
