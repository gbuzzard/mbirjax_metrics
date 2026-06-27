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
    ncu --set basic \
        --kernel-name "regex:add_fusion|dynamic_update_slice" \
        --launch-count 6 --target-processes all \
        --csv --log-file experiments/profiling/ncu/back_pixel_256.csv \
        python experiments/profiling/ncu_back_projection.py

Notes / likely iteration (ncu flags are version-sensitive — Nsight 2025.1 here):
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

# ── CONFIG ────────────────────────────────────────────────────────────────────
GEOMETRY = "cone"
SIZE = (256, 256, 256)       # match the exp-1/exp-2 GPU size; bump to 512^3 once the flow is confirmed
N_DEVICES = 1                # n=1 GPU -> the pixel kernel (the GPU-optimal back path)
WARMUP = 2                   # compile every shape BEFORE the profiled region
PROFILE_CALLS = 2            # warm calls ncu profiles (keep small — ncu replays each kernel many times)

os.environ.setdefault("MBIRJAX_NUM_CPU_DEVICES", str(N_DEVICES))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "tooling", "scaling_tests")))

import mbirjax            # noqa: E402,F401  device-setup-first
import jax                # noqa: E402
import performance_tracking as pt   # noqa: E402


def main():
    plat = jax.devices()[0].platform
    if plat != "gpu":
        print(f"  WARNING: backend is {plat}, not gpu — ncu profiling is a GPU-only step.")
    devs = jax.devices()[:N_DEVICES]
    size_label = "x".join(str(s) for s in SIZE)
    print(f"  ncu driver: {GEOMETRY} back | {size_label} | n={N_DEVICES} {plat}")

    config = pt.Config()
    model = pt.make_model(config, GEOMETRY, SIZE)
    if hasattr(model, "configure_devices"):
        model.configure_devices(devs)
    idx = pt.make_indices(model)
    sino = pt.to_device(model, pt.make_sinogram(config, SIZE), "sino")
    run_fn = lambda: model.sparse_back_project(sino, idx)

    for _ in range(WARMUP):                 # compile (ncu will also see these kernels; ignore them)
        jax.block_until_ready(run_fn())
    t0 = time.perf_counter()
    for _ in range(PROFILE_CALLS):          # the region of interest for ncu
        jax.block_until_ready(run_fn())
    dt = (time.perf_counter() - t0) / PROFILE_CALLS * 1e3
    print(f"  warm time ~{dt:.1f} ms/call  (profiled {PROFILE_CALLS} call(s))")


if __name__ == "__main__":
    main()
