"""
experiments/profiling/profile_measure.py
───────────────────────────────────────────
Produce the per-platform profiling record: for each cell (geom|op|size|ndev) capture a warm trace
AND the compiled HLO from the SAME jitted driver, join them (region_attribution) into self-time per
jax.named_scope region, and write one YAML per platform.

This is the data-production tool for the before/after-a-redesign comparison.  Design notes:
- **Trace + HLO from the same callable.** We trace the jitted projector DRIVER (not the full
  Python-orchestrated op) and lower the SAME driver for the HLO, so the trace fusions and the HLO
  fusion->region map line up.  At n=1 the sharded orchestration is ~0, so the driver IS the cost.
- **The driver depends on the execution PATH**, which is platform-specific for back projection:
  CPU uses the band kernel, GPU n=1 uses the pixel kernel (the short-circuit).  So the back cell's
  regions are ``cone/back/band/*`` on CPU vs ``cone/back/pixel/*`` on GPU — the platform divergence
  is visible, by design, and is why we keep one file PER PLATFORM.
- **Flexible schema.** Regions are discovered from the HLO; `cells[c].regions` is a dict keyed by
  whatever scopes were found.  Nothing here (or in the parser) hardcodes a region taxonomy.

Output: experiments/profiling/results/profile_<plat>_<commitUTC>_<sha8>.yaml (NOT the regression
results dir — kept separate so CPU and GPU runs never write each other's file).

Run:  python experiments/profiling/profile_measure.py
"""
import os
import sys
import glob
import time
import subprocess
import datetime as _dt

# Config lives in profiling.env (see profiling_config.py); importing it sets MBIRJAX_NUM_CPU_DEVICES.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from profiling_config import GEOMETRY, OPS, SIZE, N_DEVICES, WARMUP, TRACE_ITERS  # noqa: E402
from region_attribution import region_breakdown   # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "tooling", "scaling_tests")))

import mbirjax            # noqa: E402,F401  device-setup-first
import jax                # noqa: E402
import performance_tracking as pt   # noqa: E402
import scaling_common as sc          # noqa: E402

_RESULTS = os.path.join(_HERE, "results")   # profiling results live HERE, not in the regression dir


def _mbirjax_provenance():
    """(commit_sha, commit_date_iso) of the LOADED mbirjax checkout, best-effort."""
    d = os.path.dirname(getattr(mbirjax, "__file__", "") or "")
    def g(args):
        try:
            r = subprocess.run(["git", "-C", d, *args], capture_output=True, text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:   # noqa: BLE001
            return None
    return g(["rev-parse", "HEAD"]), g(["show", "-s", "--format=%cI", "HEAD"])


def _driver_for(pf, op, plat, sino, cyl, idx, num_slices):
    """Return (callable f, args) for the jitted projector driver on this op + execution path.

    ``f(*args)`` runs the jitted driver (trace + warm timing); ``jax.jit(f).lower(*args)`` gives the
    matching HLO.  Back projection's driver is PATH-dependent (CPU band kernel vs GPU n=1 pixel kernel).
    """
    if op == "forward":
        return (lambda c, p: pf.sparse_forward_project(c, p)), (cyl, idx)
    if op == "back":
        if plat == "gpu":
            return (lambda s, p: pf.sparse_back_project(s, p)), (sino, idx)          # pixel kernel
        return (lambda s, p: pf.sparse_back_project_band(s, p, 0, num_slices)), (sino, idx)  # band kernel
    raise ValueError(f"unsupported op {op!r}")


def _find_perfetto(out_dir):
    cands = glob.glob(os.path.join(out_dir, "**", "*.json.gz"), recursive=True)
    return next((c for c in cands if "perfetto" in os.path.basename(c).lower()), cands[0] if cands else None)


def measure_cell(model, op, plat, devs):
    """Trace + lower the matching driver, join to regions; return (cell_key, cell_record)."""
    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    idx = jax.device_put(pt.make_indices(model), devs[0])
    num_pixels, num_slices = len(idx), recon_shape[2]
    sino = jax.device_put(pt.make_sinogram(pt.Config(), SIZE), devs[0])
    cyl = jax.device_put(pt.make_cylinders(num_pixels, num_slices, 0), devs[0])
    pf = model.projector_functions
    f, args = _driver_for(pf, op, plat, sino, cyl, idx, num_slices)
    run = lambda: f(*args)

    for _ in range(WARMUP):
        jax.block_until_ready(run())
    out_dir = os.path.join(_HERE, "traces", f"prof_{GEOMETRY}_{op}_n{N_DEVICES}_{plat}")
    os.makedirs(out_dir, exist_ok=True)
    times = []
    with jax.profiler.trace(out_dir, create_perfetto_trace=True):
        for i in range(TRACE_ITERS):
            with jax.profiler.StepTraceAnnotation(op, step_num=i):
                t0 = time.perf_counter(); jax.block_until_ready(run()); times.append((time.perf_counter() - t0) * 1e3)
    hlo = jax.jit(f).lower(*args).compile().as_text()
    wall = round(min(times), 3)
    # Region 'ms' is the WALL-ATTRIBUTED share (pct x wall), NOT raw trace self-time: on CPU the
    # intra-op-thread TraceMe spans overlap, so raw self-times overcount (their sum >> wall).  The
    # SHARE (pct) is stable across that overlap, and pct x wall gives an interpretable absolute that
    # sums to ~wall on both platforms.  (On GPU the single compute stream barely overlaps, so this
    # nearly equals the raw self-time anyway.)
    raw = region_breakdown(_find_perfetto(out_dir), hlo)
    regions = {r: {"pct": v["pct"], "ms": round(v["pct"] / 100.0 * wall, 1)} for r, v in raw.items()}

    size_label = "x".join(str(s) for s in SIZE)
    key = f"{GEOMETRY}|{op}|{size_label}|{N_DEVICES}"
    rec = {"wall_ms": wall, "regions": regions}
    print(f"  {key}:  wall={wall:.1f} ms")
    for r, v in regions.items():
        print(f"      {v['pct']:5.1f}%  {v['ms']:8.1f} ms  {r}")
    return key, rec


def main():
    devs = jax.devices()[:N_DEVICES]
    plat = devs[0].platform
    sha, cdate = _mbirjax_provenance()
    try:
        import jaxlib; jaxlib_v = jaxlib.__version__
    except Exception:   # noqa: BLE001
        jaxlib_v = None
    print("=" * 78)
    print(f"  PROFILE  {GEOMETRY} {OPS} | {SIZE} | n={N_DEVICES} {plat} | jax {jax.__version__} | mbirjax {(sha or '?')[:8]}")
    print("=" * 78)

    cells = {}
    for op in OPS:
        model = pt.make_model(pt.Config(), GEOMETRY, SIZE)
        if hasattr(model, "configure_devices"):
            model.configure_devices(devs)
        key, rec = measure_cell(model, op, plat, devs)
        cells[key] = rec

    record = {
        "run": {
            "schema": 1,
            "note": "regions are DISCOVERED from jax.named_scope in the HLO; not a fixed taxonomy",
            "mbirjax_commit": sha, "mbirjax_commit_date": cdate,
            "collected": _dt.datetime.now().isoformat(timespec="seconds"),
            "platform": plat,
            "env": {"jax": jax.__version__, "jaxlib": jaxlib_v,
                    "device": devs[0].device_kind, "n_devices": N_DEVICES},
        },
        "cells": cells,
    }
    stamp = (_dt.datetime.fromisoformat(cdate).astimezone(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
             if cdate else _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(_RESULTS, exist_ok=True)
    path = os.path.join(_RESULTS, f"profile_{plat}_{stamp}_{(sha or 'nosha')[:8]}.yaml")
    sc.save_yaml(path, record)


if __name__ == "__main__":
    main()
