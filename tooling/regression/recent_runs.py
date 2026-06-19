#!/usr/bin/env python3
"""Tile-style summary of recent nightly runs (newest first).

DRY: this does NOT re-parse the result YAML.  It calls the dashboard's own
``build_dashboard.collect_data()`` — so platform/branch/commit, configs, gate hits and tests come
from the exact same code (and the exact same record shape) the dashboard renders.  The only thing
added here is the run-level thermal flag, which mirrors cellHot()/cellThrottled()/runThermal() in
dashboard.js (JS and Python can't share a runtime, so the small rule lives once on each side).

Usage:  recent_runs.py <results_repo_root> [N]
build_dashboard is imported from THIS checkout's tooling/viewer (so the slim-cell shape matches the
thermal rule below), then its REPO_ROOT is pointed at <results_repo_root> so collect_data() reads
THAT repo's results/ — e.g. the nightly's persistent clone, even when the clone's own (pushed)
build_dashboard predates the thermal fields.  Requires the same dependency the dashboard build needs
(PyYAML, in the mbirjax/harness conda env); exits 3 if build_dashboard can't be imported/run so the
caller can fall back to a filename listing.
"""
import sys
import os
from pathlib import Path
from datetime import datetime

HOT_C, HOT_HBM = 85, 95  # mirror dashboard.js cellHot()


def _throttled(c):   # causal: a driver throttle reason fired (slim gpu key "thr")
    return any(g.get("thr") for g in (c.get("gpu") or []))


def _hot(c):         # advisory superset of throttled (slim keys: t=core°C, mt=HBM°C)
    if c.get("throttled") or _throttled(c):
        return True
    return any((g.get("t") or 0) >= HOT_C or (g.get("mt") or 0) >= HOT_HBM for g in (c.get("gpu") or []))


def _thermal(cells):
    """Worst severity over the run, the device counts (ndev) it hit, and the peak core temp."""
    acc = {"throttled": {"d": set(), "t": 0}, "hot": {"d": set(), "t": 0}}
    for c in cells:
        sev = "throttled" if _throttled(c) else ("hot" if _hot(c) else None)
        if not sev:
            continue
        if c.get("ndev") is not None:
            acc[sev]["d"].add(c["ndev"])
        peak = max([0] + [g.get("t") or 0 for g in (c.get("gpu") or [])])
        acc[sev]["t"] = max(acc[sev]["t"], peak)
    for sev in ("throttled", "hot"):
        if acc[sev]["d"]:
            return sev, sorted(acc[sev]["d"]), acc[sev]["t"]
    return None


def _commit_minute(run):
    s = run.get("commit_date")
    if not s:
        return run.get("date") or "?"
    try:
        return datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return s[:16].replace("T", " ")


def main():
    if len(sys.argv) < 2:
        print("usage: recent_runs.py <metrics_repo_root> [N]", file=sys.stderr)
        sys.exit(2)
    root = os.path.abspath(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    # Use THIS checkout's build_dashboard (its slim-cell shape matches _thermal below), but read the
    # target repo's results/ by overriding REPO_ROOT.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "viewer"))
    try:
        import build_dashboard
        build_dashboard.REPO_ROOT = Path(root)
        runs = build_dashboard.collect_data().get("runs", [])
    except Exception as e:                       # noqa: BLE001 — status tool: degrade, never abort
        print(f"  (could not load runs via build_dashboard for {root}: {e})", file=sys.stderr)
        sys.exit(3)
    if not runs:
        print(f"  (no runs under {root}/results)")
        return

    runs.sort(key=lambda r: (r.get("commit_date") or "", r.get("date") or ""), reverse=True)
    print(f"recent runs (from {root}/results, newest first):")
    for r in runs[:n]:
        cells = r.get("cells") or []
        nfail = sum(1 for c in cells if c.get("failed"))
        hard = len((r.get("gate") or {}).get("hard") or [])
        tests = r.get("tests")
        tf = tests.get("failed") if tests else None
        therm = _thermal(cells)

        cellstr = f"cells {len(cells)}" + (f" ({nfail} failed)" if nfail else "")
        teststr = "tests " + ("–" if tf is None else str(tf))
        warn = ""
        if therm:
            sev, devs, peak = therm
            word = "throttled" if sev == "throttled" else "ran hot"
            warn = f"  ⚠ {word} · n={','.join(map(str, devs))}" + (f" · up to {peak}°C" if peak else "")
        glyph = "✗" if (nfail or hard or (tf or 0)) else ("⚠" if therm else "✓")
        dirty = " ·dirty" if r.get("dirty") else ""
        print(f"  {glyph} {_commit_minute(r)}  {r.get('platform', '?'):<3} {r.get('branch', '?')} "
              f"@ {r.get('commit', '')[:8]}{dirty}  {cellstr} · gate {hard} · {teststr}{warn}")


if __name__ == "__main__":
    main()
