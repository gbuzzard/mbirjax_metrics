#!/usr/bin/env python3
"""Ad-hoc: measure ONE nightly cell group, e.g.

    === parallel | forward | 200x208x160 @ n=[1, 2, 4] ===

Reproduces exactly what the nightly measures for a single (geometry, op, size) -- same model build,
inputs, sharding, warmup, and timing loop -- by calling the engine's own
``performance_tracking.measure_cell_group`` (so the numbers are comparable to the regression YAMLs),
but for just that one cell so a slowdown can be bisected without a full run.

It measures whatever mbirjax is importable in the CURRENT environment (no worktree/PYTHONPATH magic),
so to bisect: `pip install -e` (or PYTHONPATH) the commit under test, then run this.  Single-process
(inline) for easy debugging -- peak memory is read live, so it may differ slightly from the nightly's
per-config subprocess isolation; the TIMING is what this is for.

Usage:
    python measure_one_cell.py                              # parallel | forward | 200x208x160 @ [1,2,4]
    python measure_one_cell.py --op back --size 512x448x384
    python measure_one_cell.py --geometry cone --device-counts 4 2 1 --trials 5
"""
import argparse
import os
import sys
import tempfile

import performance_tracking as pt   # sibling module; this script's dir is on sys.path[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", default="parallel")
    ap.add_argument("--op", default="forward",
                    help="direct_filter | forward | back | vcd_nonconst | denoise")
    ap.add_argument("--size", default="200x208x160", help="e.g. 200x208x160 (denoiser: image shape)")
    ap.add_argument("--device-counts", type=int, nargs="+", default=[1, 2, 4],
                    help="device counts to sweep (run_measure_loop descends through them, OOM-aware)")
    ap.add_argument("--trials", type=int, default=None, help="override trials/op (e.g. 1 or 5)")
    ap.add_argument("--warmup", type=int, default=None, help="override warmup iterations")
    args = ap.parse_args()

    config = pt.Config(inline=True)
    if args.trials is not None:
        config.trials_by_op = {k: args.trials for k in config.trials_by_op}
        config.single_trial_sizes = []        # don't let the 1024 single-trial rule override --trials
    if args.warmup is not None:
        config.warmup = args.warmup

    dc = sorted(set(args.device_counts))
    print(f"\n=== {args.geometry} | {args.op} | {args.size} @ n={dc} ===")
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="perf_adhoc_")
    os.close(fd)
    try:
        res = pt.measure_cell_group(config, args.geometry, args.op, args.size, dc, tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    rows = res.get("rows") or []
    pt.sc.annotate_speedups(rows)             # 'speedup' vs the fewest-device run, as the nightly does
    print(f"\nsummary ({args.geometry} | {args.op} | {args.size}):")
    for r in sorted(rows, key=lambda r: r["n_devices"]):
        mn, mem, sp = r.get("min_ms"), r.get("mem_mb"), r.get("speedup", 1.0)
        print(f"  n={r['n_devices']}   min={mn:9.1f} ms   mem={mem:9.1f} MB   speedup={sp:.2f}x")
    for f in res.get("failures") or []:
        print(f"  n={f['n_devices']}   FAILED{' (OOM)' if f.get('oom') else ''}: {f.get('error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
