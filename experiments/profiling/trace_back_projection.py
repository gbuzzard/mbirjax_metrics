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
import gzip
import json
import glob
import time
from collections import defaultdict
from datetime import datetime

# ── CONFIG (edit here) ────────────────────────────────────────────────────────
GEOMETRY = "cone"            # back projection's most-analyzed geometry (band path; cone cliff lives here)
# SINOGRAM size (n_views, n_rows, n_channels); the cone recon is auto-derived.  256-class as agreed.
# (The nightly uses ASYMMETRIC sizes to surface axis swaps; symmetry is fine for a profiling dry run.)
SIZE = (256, 256, 256)
N_DEVICES_LIST = [1, 2]      # device counts to trace, in order.  GPU: n=1 short-circuits to the pixel
                             # kernel; n>=2 exercises the banded reduce-scatter (NVLink).  Counts above the
                             # number of available devices are skipped with a note.
WARMUP = 2                   # untimed calls to trigger compilation of every band/batch shape
TRACE_ITERS = 3              # warm iterations captured in the trace (and timed)
TOP_N = 30                   # how many trace events to print in the summary table

# ── Device-setup-first: choose the CPU device count BEFORE importing mbirjax ──
# mbirjax reads MBIRJAX_NUM_CPU_DEVICES on its first import to size the virtual CPU
# device mesh, so it must be set before `import mbirjax`.  setdefault respects a value
# already set in the shell/cluster.
os.environ.setdefault("MBIRJAX_NUM_CPU_DEVICES", str(max(N_DEVICES_LIST)))

# Make the engine's helpers importable (they live next to the nightly engine).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCALING = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "tooling", "scaling_tests"))
sys.path.insert(0, _SCALING)

import mbirjax            # noqa: E402,F401 — device-setup side effect; must precede `import jax`
import jax                # noqa: E402
import performance_tracking as pt   # noqa: E402  (reuses make_model/make_sinogram/... — the SAME inputs as the nightly)


def _is_host_runtime(name):
    """True for host-side Python frames / runtime wrappers (vs an XLA fusion/op).

    These are wait/dispatch/orchestration events (`$api.py:... block_until_ready`, the worker
    thread `_bootstrap`, `ThunkExecutor::Execute`, `SlinkyThreadPool::Await`, our own
    StepTraceAnnotation), NOT compute.  We bucket them separately so the compute fusions
    (bitcast_gather_fusion, broadcast_atan2_fusion, ...) stand out.
    """
    return (name.startswith("$") or "::" in name or ".py:" in name
            or name in ("back_project",) or name.startswith("end:"))


def summarize_perfetto(trace_path, n_iters, top_n=TOP_N):
    """Self-time (exclusive) summary of a Perfetto/Chrome-trace JSON.

    The trace is the Chrome Trace Event format: ph='X' complete events with ts/dur
    (microseconds) and pid/tid identifying a timeline track.  Within one track the events
    strictly nest, so we compute SELF-TIME = dur - sum(direct children dur) with a stack
    sweep.  Self-time avoids the trap of the naive sum, where wrapper events (the whole-op
    StepTraceAnnotation, block_until_ready, the worker thread's lifetime) dominate because
    they CONTAIN everything.

    Three views, most useful first:
      * by TRACK — where the wall time lived (the host:CPU dispatch track vs the tf_XLAEigen
        intra-op worker threads that do the real compute).
      * by FUSION FAMILY — the .N variants of each XLA fusion merged (the compute op classes).
      * by EVENT NAME — the raw leaderboard (host-runtime waits included, for completeness).

    NOTE (honest caveat): on the CPU backend a fusion's TraceMe span covers dispatch+execute
    and can overlap the worker threads, so the per-fusion seconds are a reliable RANKING but
    not exact exclusive time; the TRACK view is the truer compute/wait split.
    """
    with gzip.open(trace_path, "rt") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    pname, tname = {}, {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pname[e.get("pid")] = e.get("args", {}).get("name", "")
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            tname[(e.get("pid"), e.get("tid"))] = e.get("args", {}).get("name", "")

    by_tid = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and e.get("dur") is not None:
            by_tid[(e.get("pid"), e.get("tid"))].append(e)

    self_us = defaultdict(lambda: [0.0, 0])   # name -> [self_us, count]
    track_self = defaultdict(float)           # track label -> self_us
    for key, evs in by_tid.items():
        evs.sort(key=lambda e: (e["ts"], -e["dur"]))   # parent before child on ties
        label = tname.get(key, pname.get(key[0], "?"))
        stack = []   # [ts, end, name, child_total_us]
        def _pop(s):
            st = max(0.0, (s[1] - s[0]) - s[3])
            self_us[s[2]][0] += st; self_us[s[2]][1] += 1
            track_self[label] += st
        for e in evs:
            ts, dur = float(e["ts"]), float(e["dur"]); end = ts + dur
            while stack and stack[-1][1] <= ts:
                _pop(stack.pop())
            if stack:
                stack[-1][3] += dur
            stack.append([ts, end, e.get("name", "?"), 0.0])
        while stack:
            _pop(stack.pop())

    n = max(n_iters, 1)
    print(f"\n  trace events: {len(events)} total   (self-time, exclusive; per-iter = /{n})")

    print(f"\n  === SELF-TIME by TRACK (compute threads vs host/dispatch) ===")
    print(f"  {'self_ms':>10}  {'/iter':>8}  track")
    print("  " + "-" * 60)
    for lab, us in sorted(track_self.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {us / 1e3:>10.1f}  {us / 1e3 / n:>8.1f}  {lab[:48]}")

    fam = defaultdict(lambda: [0.0, 0])
    for name, (us, cnt) in self_us.items():
        if _is_host_runtime(name):
            key = "[host/runtime] " + (name.split(":")[0] if ".py:" in name else name.split("::")[0])[:28]
        else:
            tail = name.rsplit(".", 1)[-1]
            key = name.rsplit(".", 1)[0] if tail.isdigit() else name
        fam[key][0] += us; fam[key][1] += cnt
    print(f"\n  === SELF-TIME by FUSION FAMILY (xla compute; .N variants merged) ===")
    print(f"  {'self_ms':>10}  {'/iter':>8}  name")
    print("  " + "-" * 60)
    for name, (us, cnt) in sorted(fam.items(), key=lambda kv: -kv[1][0])[:top_n]:
        print(f"  {us / 1e3:>10.1f}  {us / 1e3 / n:>8.1f}  {name[:48]}")
    return sorted(self_us.items(), key=lambda kv: -kv[1][0])


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
