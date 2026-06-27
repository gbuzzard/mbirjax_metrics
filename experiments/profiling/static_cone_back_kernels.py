"""
experiments/profiling/static_cone_back_kernels.py
──────────────────────────────────────────────────
Experiment 2 of the projector-profiling investigation: attach STATIC analysis
(`cost_analysis` / `memory_analysis` / HLO text) to the actual jitted back-projection
KERNELS, and use it as a known-answer check on the documented cone CPU "cache cliff".

Background (from mbirjax/.claude/lessons.md, "platform-divergent back kernel"):
cone has TWO back kernels with OPPOSITE platform rankings.

  * pixel kernel  (back_project_one_view_to_pixel_batch, via projector_functions.sparse_back_project):
        a rolled `lax.map` + transpose.  UNDER the view-vmap this is a FUSION BARRIER, so XLA
        materializes the full (views x npix x slices) per-view stack -> CPU cache-thrash at >=~200^3
        (measured x62/x110), while it is the FASTER kernel on GPU.
  * band kernel   (back_project_one_view_to_band, via projector_functions.sparse_back_project_band):
        no lax.map/transpose, keeps vmap+sum fused -> no materialization -> ~8x FASTER on CPU.

This is the smallest faithful isolation: we call the two jitted DRIVERS directly (no sharded
thread-pool / reduce-scatter around them) at one size BELOW the cliff (128^3) and one ABOVE
(256^3), and for each we print:
  * warm wall time                          (the cliff should show as pixel >> band at 256^3)
  * memory_analysis temp bytes              (the materialized stack should make pixel's temp huge)
  * cost_analysis flops / bytes accessed    (compute vs memory traffic)
  * the compiled HLO dumped to a .txt       (to read the fusion structure by eye)

If the static numbers + warm timing reproduce the known cliff, we trust this tooling for the
GPU questions it can't answer locally; if not, we've learned its resolution limits cheaply.

Run (repo root, `mbirjax` conda env):  python experiments/profiling/static_cone_back_kernels.py
"""
import os
import sys
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
GEOMETRY = "cone"
SIZES = [(128, 128, 128), (256, 256, 256)]   # below / above the ~200^3 cliff
WARMUP = 1
TIMED = 3
N_DEVICES = 1                                 # single device: isolate the KERNEL, not sharding
DUMP_HLO = True

os.environ.setdefault("MBIRJAX_NUM_CPU_DEVICES", str(N_DEVICES))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "tooling", "scaling_tests")))

import mbirjax            # noqa: E402,F401  device-setup-first
import jax                # noqa: E402
import performance_tracking as pt   # noqa: E402
import scaling_common as sc          # noqa: E402  (YAML writer for the off-box / Samba workflow)


def _mem(compiled):
    """Pull the bytes breakdown from a compiled executable's memory_analysis (defensive: the
    attribute set varies by jaxlib).  Returns a dict of MB for temp / argument / output."""
    try:
        ma = compiled.memory_analysis()
    except Exception as e:   # noqa: BLE001
        return {"error": str(e)[:80]}
    out = {}
    for k in ("temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes",
              "generated_code_size_in_bytes"):
        v = getattr(ma, k, None)
        if v is not None:
            out[k.replace("_size_in_bytes", "")] = v / (1024 ** 2)   # MB
    return out


def _cost(compiled):
    """FLOPs and bytes-accessed from cost_analysis (a dict, or list of dicts)."""
    try:
        ca = compiled.cost_analysis()
    except Exception as e:   # noqa: BLE001
        return {"error": str(e)[:80]}
    if isinstance(ca, (list, tuple)):
        ca = ca[0] if ca else {}
    return {"gflops": ca.get("flops", 0) / 1e9,
            "gbytes_accessed": ca.get("bytes accessed", 0) / 1e9}


def analyze(label, fn, args, size_label):
    """Lower+compile `fn(*args)`, print static cost/memory, time it warm, dump HLO."""
    compiled = jax.jit(fn).lower(*args).compile()
    mem, cost = _mem(compiled), _cost(compiled)
    # Warm timing (compile already done above is a separate cache; warm the call cache too).
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - t0) * 1e3)
    tmin = min(times)
    print(f"    {label:<8s}  time(min)={tmin:8.1f} ms   "
          f"temp={mem.get('temp', float('nan')):8.1f} MB   "
          f"out={mem.get('output', float('nan')):7.1f} MB   "
          f"GFLOP={cost.get('gflops', float('nan')):7.1f}   "
          f"GB_acc={cost.get('gbytes_accessed', float('nan')):7.1f}")
    if DUMP_HLO:
        hlo_path = os.path.join(_HERE, "hlo", f"{GEOMETRY}_{label}_{size_label}.txt")
        os.makedirs(os.path.dirname(hlo_path), exist_ok=True)
        with open(hlo_path, "w") as f:
            f.write(compiled.as_text())
    return {"size": size_label, "kernel": label, "time_min_ms": tmin,
            "temp_mb": mem.get("temp"), "out_mb": mem.get("output"),
            "gflops": cost.get("gflops"), "gbytes_accessed": cost.get("gbytes_accessed")}


def main():
    plat = jax.devices()[0].platform
    print("=" * 90)
    print(f"  STATIC cone back kernels (pixel vs band) | n={N_DEVICES} {plat} | jax {jax.__version__}")
    print("=" * 90)
    config = pt.Config()
    rows = []
    for size in SIZES:
        size_label = "x".join(str(s) for s in size)
        model = pt.make_model(config, GEOMETRY, size)
        if hasattr(model, "configure_devices"):
            model.configure_devices(jax.devices()[:N_DEVICES])
        recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
        idx = pt.make_indices(model)
        num_pixels, num_slices = len(idx), recon_shape[2]
        sino = pt.to_device(model, pt.make_sinogram(config, size), "sino")
        idx = jax.device_put(idx, jax.devices()[0])

        pf = model.projector_functions
        # pixel-kernel driver (the cliff suspect) and band-kernel driver (the cliff-avoider).
        pixel_fn = lambda s, p: pf.sparse_back_project(s, p)
        band_fn = lambda s, p: pf.sparse_back_project_band(s, p, 0, num_slices)
        # Expected materialized stack if XLA can't fuse the pixel path: views*npix*slices*4 bytes.
        stack_mb = size[0] * num_pixels * num_slices * 4 / (1024 ** 2)

        print(f"\n  size={size_label}  recon={recon_shape}  num_pixels={num_pixels}  "
              f"num_slices={num_slices}   (full view*npix*slices stack = {stack_mb:,.0f} MB)")
        pr = analyze("pixel", pixel_fn, (sino, idx), size_label)
        br = analyze("band", band_fn, (sino, idx), size_label)
        rows += [pr, br]
        print(f"    -> pixel/band time ratio = {pr['time_min_ms'] / br['time_min_ms']:.2f}x   "
              f"(lesson: CPU ~1x below the ~200^3 cliff & ~8x above; GPU EXPECT <1x — pixel faster)")

    plat = jax.devices()[0].platform
    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"static_cone_back_{plat}.yaml"),
                 {"platform": plat, "geometry": GEOMETRY, "jax": jax.__version__, "rows": rows})


if __name__ == "__main__":
    main()
