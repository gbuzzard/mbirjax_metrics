"""
experiments/profiling/trace_utils.py
───────────────────────────────────────
Shared, JAX-FREE helpers for the trace scripts (back / forward / parallel / qGGMRF): parse a
Perfetto/Chrome-trace JSON and print a SELF-TIME (exclusive) summary.  Pure stdlib so importing
it has no device-setup side effects — the per-op trace scripts own the mbirjax/jax imports.

Self-time = dur - sum(direct children dur), computed with a per-track stack sweep.  It avoids the
trap of a naive per-name sum, where wrapper events (the whole-op StepTraceAnnotation,
block_until_ready, a worker thread's lifetime) dominate because they CONTAIN everything.

Honest caveat: on the CPU backend a fusion's TraceMe span covers dispatch+execute and can overlap
the worker threads, so the per-fusion seconds are a reliable RANKING, not exact exclusive time; the
TRACK view is the truer compute/wait split.  On GPU the device streams give per-CUDA-kernel times.
"""
import gzip
import json
from collections import defaultdict


def is_host_runtime(name):
    """True for host-side Python frames / runtime wrappers (vs an XLA fusion / CUDA kernel).

    These are wait/dispatch/orchestration events (`$api.py:... block_until_ready`, a worker thread
    `_bootstrap`, `ThunkExecutor::Execute`, `SlinkyThreadPool::Await`, the StepTraceAnnotation), NOT
    compute, so we bucket them separately and let the real fusions/kernels stand out.
    """
    return (name.startswith("$") or "::" in name or ".py:" in name
            or name.startswith("end:") or name in ("back_project", "forward_project"))


def summarize_perfetto(trace_path, n_iters, top_n=30):
    """Self-time summary of a Perfetto/Chrome-trace JSON: by TRACK, then by FUSION/KERNEL FAMILY.

    Args:
        trace_path: path to the `perfetto_trace.json.gz` written by `jax.profiler.trace`.
        n_iters: number of traced iterations (the per-iter columns divide by this).
        top_n: rows to print in the family table.

    Returns the self-time leaderboard (list of (name, [self_us, count])), heaviest first.
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

    print(f"\n  === SELF-TIME by TRACK (compute threads/streams vs host/dispatch) ===")
    print(f"  {'self_ms':>10}  {'/iter':>8}  track")
    print("  " + "-" * 60)
    for lab, us in sorted(track_self.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {us / 1e3:>10.1f}  {us / 1e3 / n:>8.1f}  {lab[:48]}")

    fam = defaultdict(lambda: [0.0, 0])
    for name, (us, cnt) in self_us.items():
        if is_host_runtime(name):
            key = "[host/runtime] " + (name.split(":")[0] if ".py:" in name else name.split("::")[0])[:28]
        else:
            tail = name.rsplit(".", 1)[-1]
            key = name.rsplit(".", 1)[0] if tail.isdigit() else name
        fam[key][0] += us; fam[key][1] += cnt
    print(f"\n  === SELF-TIME by FUSION/KERNEL FAMILY (.N variants merged) ===")
    print(f"  {'self_ms':>10}  {'/iter':>8}  name")
    print("  " + "-" * 60)
    for name, (us, cnt) in sorted(fam.items(), key=lambda kv: -kv[1][0])[:top_n]:
        print(f"  {us / 1e3:>10.1f}  {us / 1e3 / n:>8.1f}  {name[:48]}")
    return sorted(self_us.items(), key=lambda kv: -kv[1][0])
