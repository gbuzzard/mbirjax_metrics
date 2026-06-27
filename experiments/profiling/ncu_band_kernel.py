"""
experiments/profiling/ncu_band_kernel.py
──────────────────────────────────────────
Experiment 4b (GPU only): ncu the BAND back-projection kernel in isolation, to characterize
the bottleneck that makes multi-GPU back NON-MONOTONIC.

The n=2 trace (exp 1) showed: at n>=2 the back path drops the pixel short-circuit and runs the
band kernel, whose **`input_transpose_fusion`** dominates (~91 ms/GPU vs n=1 pixel's 69 ms), while
the NVLink reduce-scatter is cheap (~3.5 ms).  So the band kernel's transpose — not comms — is what
makes 2 GPUs slower than 1 (the "B4.5 lever").  The n=1 ncu (ncu_back_projection.py) profiled the
PIXEL kernel; this profiles the BAND kernel.

Isolation trick: call ``projector_functions.sparse_back_project_band(...)`` directly.  That is the
band driver, which has NO n=1 pixel short-circuit (the short-circuit lives in
``model.sparse_back_project``), so it runs the band kernel on a SINGLE GPU — same kernel as the
multi-device path, without the reduce-scatter / 2-device profiling complexity.

Run UNDER ncu (cluster, GPU env).  Target the transpose fusion:

    mkdir -p experiments/profiling/ncu
    ncu --profile-from-start off --set basic \
        --kernel-name "regex:transpose_fusion|input_transpose" \
        --launch-count 6 --target-processes all \
        --csv --log-file experiments/profiling/ncu/back_band_256.csv \
        python experiments/profiling/ncu_band_kernel.py

Same flag notes as ncu_back_projection.py (ERR_NVGPUCTRPERM = counters locked; drop --kernel-name
to profile the first launches if the regex misses; --set basic ~ SpeedOfLight).  Roofline read:
Memory% >> Compute% near peak DRAM => bandwidth-bound; both low => latency/occupancy-bound (headroom).
"""
import os
import sys
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
GEOMETRY = "cone"
SIZE = (256, 256, 256)       # match exp 1/2; the band kernel's transpose is the multi-GPU limiter here
N_DEVICES = 1                # isolate the band kernel on ONE GPU (no reduce-scatter)
WARMUP = 2
PROFILE_CALLS = 2

os.environ.setdefault("MBIRJAX_NUM_CPU_DEVICES", str(N_DEVICES))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "tooling", "scaling_tests")))
sys.path.insert(0, _HERE)                 # so cuda_profiler is importable
from cuda_profiler import profiler_range   # noqa: E402

import mbirjax            # noqa: E402,F401  device-setup-first
import jax                # noqa: E402
import performance_tracking as pt   # noqa: E402


def main():
    plat = jax.devices()[0].platform
    if plat != "gpu":
        print(f"  WARNING: backend is {plat}, not gpu — ncu profiling is a GPU-only step.")
    devs = jax.devices()[:N_DEVICES]
    size_label = "x".join(str(s) for s in SIZE)

    config = pt.Config()
    model = pt.make_model(config, GEOMETRY, SIZE)
    if hasattr(model, "configure_devices"):
        model.configure_devices(devs)
    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    num_slices = recon_shape[2]
    idx = jax.device_put(pt.make_indices(model), devs[0])
    sino = pt.to_device(model, pt.make_sinogram(config, SIZE), "sino")

    pf = model.projector_functions
    if not hasattr(pf, "sparse_back_project_band"):
        print(f"  ERROR: {GEOMETRY} has no band kernel (sparse_back_project_band) — nothing to profile.")
        return
    # The band driver over the FULL slice range [0, num_slices) — the band kernel, single GPU, no short-circuit.
    run_fn = lambda: pf.sparse_back_project_band(sino, idx, 0, num_slices)
    print(f"  ncu BAND driver: {GEOMETRY} back | {size_label} | n={N_DEVICES} {plat} | num_slices={num_slices}")

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
