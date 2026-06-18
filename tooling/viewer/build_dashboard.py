#!/usr/bin/env python3
"""Build a self-contained HTML dashboard from the mbirjax_metrics YAML time series.

This is a *reader/generator*: it walks the metrics repo's result files, parses them
into a compact JSON blob, and inlines that JSON — together with the page's CSS, JS,
and a vendored copy of the uPlot charting library — into a single static
``index.html``.  The output opens straight from ``file://`` (or via GitHub Pages),
needs no server, and makes no network calls.

What it reads (all already produced by the nightly engine):
  results/<plat>/<branch>/regression_<plat>_<date>.yaml   the per-run cells + gate
  results/<plat>/<branch>/records_<plat>.yaml             best-ever per cell+metric (records panel)
  results/<plat>/<branch>/tests_<plat>_<date>.txt         pytest summary line
The "compare against" overlays (main / prerelease / prior run / best-ever) are all derived from the
runs above — there is no separate reference file.

Run it on demand to refresh the page::

    cd tooling/viewer
    python build_dashboard.py        # -> <repo>/dashboard/index.html

There are no command-line arguments by design; the few knobs live in the CONFIG
block below so a run is reproducible from the checked-in source.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import subprocess
from pathlib import Path

import yaml

# --------------------------------------------------------------------------- #
# CONFIG — edit here, not via the command line.                               #
# --------------------------------------------------------------------------- #
# Repo root is inferred as two levels up from this file (tooling/viewer/ -> repo).
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
OUT_PATH = REPO_ROOT / "dashboard" / "index.html"

# Optional filters; None means "everything discovered under results/".
ONLY_PLATFORMS: list[str] | None = None      # e.g. ["cpu", "gpu"]
ONLY_BRANCHES: list[str] | None = None        # e.g. ["greg/conebeam_sharding"] (display names)

# Fingerprint fields surfaced in the per-cell drill-down (samples dropped to keep
# the embedded JSON lean as the history grows — these five summarise the array).
_FP_KEYS = ("sum", "mean", "l2norm", "min", "max")


def _repo_url() -> str | None:
    """Browsable URL of this repo's `origin` remote (for the title link), or None if unavailable.

    Normalises both forms git emits: ``git@github.com:org/repo.git`` and
    ``https://github.com/org/repo.git`` -> ``https://github.com/org/repo``.
    """
    try:
        url = subprocess.run(["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None
    if not url:
        return None
    if url.startswith("git@"):                       # git@host:org/repo(.git)
        url = "https://" + url[4:].replace(":", "/", 1)
    return url[:-4] if url.endswith(".git") else url


# --------------------------------------------------------------------------- #
# Parsing helpers                                                             #
# --------------------------------------------------------------------------- #
def _slim_fingerprint(fp: dict | None) -> dict | None:
    """Keep the interpretable scalar summary; drop the raw sample list."""
    if not fp:
        return None
    out = {k: fp.get(k) for k in _FP_KEYS}
    if fp.get("shape") is not None:
        out["shape"] = fp["shape"]
    if fp.get("dtype") is not None:
        out["dtype"] = fp["dtype"]
    return out


def _slim_cell(c: dict) -> dict:
    """Reduce a raw regression cell to the fields the page needs."""
    return {
        "geom": c.get("geometry"),
        "op": c.get("op"),
        "size": c.get("size"),
        "ndev": c.get("n_devices"),
        "min_ms": c.get("min_ms"),
        "mem_mb": c.get("mem_mb"),
        "speedup": c.get("speedup"),
        "is_sharded": c.get("is_sharded"),
        "throttled": c.get("throttled"),
        "failed": bool(c.get("failed", False)),
        "oom": bool(c.get("oom", False)),
        "error": c.get("error"),
        "fp": _slim_fingerprint(c.get("fingerprint")),
    }


_TESTS_RE = {
    "passed": re.compile(r"(\d+)\s+passed"),
    "failed": re.compile(r"(\d+)\s+failed"),
    "skipped": re.compile(r"(\d+)\s+skipped"),
}


def _parse_tests(txt_path: Path) -> dict | None:
    """Pull pass/fail/skip counts and the failing node-ids from a pytest log."""
    if not txt_path.exists():
        return None
    text = txt_path.read_text(errors="replace")
    out = {}
    for kind, rx in _TESTS_RE.items():
        m = rx.search(text)
        out[kind] = int(m.group(1)) if m else 0
    # The "short test summary info" block lists one `FAILED <nodeid>[ - <err>]`
    # line per failure; capture the node-ids for the tile drill-down.
    failures = []
    for line in text.splitlines():
        if line.startswith("FAILED "):
            failures.append(line[len("FAILED "):].split(" - ", 1)[0].strip())
    if not (out["passed"] or out["failed"] or out["skipped"]):
        return None
    out["failures"] = failures
    return out


_GATE_BASIS_RE = re.compile(r"^\[(\w+)\]\s*")
_GATE_CELL_RE = re.compile(r"([a-z_]+\|[a-z_]+\|\d+x\d+x\d+\|\d+)")


def _parse_gate_hard(items) -> list[dict]:
    """Split each hard-gate string into {basis, cell, text}.

    The engine prefixes each entry with its comparison basis, e.g.
    ``[prior:regression_gpu_...yaml] cone|back|512x448x384|1 memory ...`` — so the dashboard can
    show what it was compared against and mark the offending cell on the plot.
    """
    out = []
    for s in items or []:
        s = str(s)
        mb = _GATE_BASIS_RE.match(s)
        mc = _GATE_CELL_RE.search(s)
        out.append({
            "basis": mb.group(1) if mb else None,
            "cell": mc.group(1) if mc else None,
            "text": _GATE_BASIS_RE.sub("", s).strip(),
        })
    return out


def _parse_run(path: Path, platform: str, branch_dir: str) -> dict:
    """Parse one regression_<plat>_<date>.yaml into a compact run record."""
    doc = yaml.safe_load(path.read_text()) or {}
    date = str(doc.get("date") or "")
    branch = doc.get("git_branch") or branch_dir.replace("_", "/")
    gate = doc.get("gate") or {}
    cfg = doc.get("config") or {}
    tests = _parse_tests(path.parent / f"tests_{platform}_{date}.txt")
    return {
        "platform": platform,
        "branch": branch,
        "branch_dir": branch_dir,
        "date": date,
        "commit_date": doc.get("git_commit_date"),  # ISO commit time (None on older runs)
        "commit": (doc.get("git_commit") or "")[:10],
        "commit_full": doc.get("git_commit") or "",
        "version": doc.get("mbirjax_version"),
        "dirty": bool(doc.get("git_dirty", False)),
        "device_label": doc.get("device_label"),
        "gate": {
            "result": gate.get("result"),
            "hard": _parse_gate_hard(gate.get("hard")),
            "soft": [str(s) for s in (gate.get("soft") or [])],
            "compared_to": gate.get("compared_to") or [],
        },
        # thresholds the gate uses, surfaced in the gate-tile explanation
        "gate_config": {k: cfg.get(k) for k in
                        ("mem_hard_pct", "speedup_warn_pct", "time_soft_pct",
                         "fp_rtol_single", "fp_rtol_iter")},
        "tests": tests,
        "cells": [_slim_cell(c) for c in (doc.get("cells") or [])],
    }


# --------------------------------------------------------------------------- #
# Discovery                                                                   #
# --------------------------------------------------------------------------- #
def collect_data() -> dict:
    results_dir = REPO_ROOT / "results"

    runs: list[dict] = []
    records: dict[str, dict] = {}
    platforms: set[str] = set()
    branches: set[str] = set()

    for plat_dir in sorted(p for p in results_dir.glob("*") if p.is_dir()):
        platform = plat_dir.name
        if ONLY_PLATFORMS and platform not in ONLY_PLATFORMS:
            continue
        for branch_dir_path in sorted(p for p in plat_dir.glob("*") if p.is_dir()):
            branch_dir = branch_dir_path.name
            run_files = sorted(branch_dir_path.glob(f"regression_{platform}_*.yaml"))
            if not run_files:
                continue
            for rf in run_files:
                run = _parse_run(rf, platform, branch_dir)
                if ONLY_BRANCHES and run["branch"] not in ONLY_BRANCHES:
                    continue
                runs.append(run)
                platforms.add(platform)
                branches.add(run["branch"])
            # Records book (best-ever) — one per platform+branch.
            rec_path = branch_dir_path / f"records_{platform}.yaml"
            if rec_path.exists():
                branch_name = runs[-1]["branch"] if runs else branch_dir.replace("_", "/")
                records[f"{platform}|{branch_name}"] = yaml.safe_load(rec_path.read_text()) or {}

    runs.sort(key=lambda r: (r["platform"], r["branch"], r["date"]))
    return {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "repo_name": REPO_ROOT.name,
        "repo_url": _repo_url(),
        "platforms": sorted(platforms),
        "branches": sorted(branches),
        "runs": runs,
        "records": records,
    }


# --------------------------------------------------------------------------- #
# Assembly                                                                    #
# --------------------------------------------------------------------------- #
def build() -> Path:
    data = collect_data()

    template = (HERE / "template.html").read_text()
    css = (HERE / "dashboard.css").read_text()
    js = (HERE / "dashboard.js").read_text()
    uplot_js = (HERE / "vendor" / "uPlot.iife.min.js").read_text()
    uplot_css = (HERE / "vendor" / "uPlot.min.css").read_text()

    # json.dumps is safe to inline as long as we neutralise the only sequence that
    # can prematurely close a <script> element.
    data_json = json.dumps(data, allow_nan=False, default=str).replace("</", "<\\/")

    html = (
        template
        .replace("/*{{UPLOT_CSS}}*/", uplot_css)
        .replace("/*{{DASHBOARD_CSS}}*/", css)
        .replace("/*{{UPLOT_JS}}*/", uplot_js)
        .replace("/*{{DATA}}*/", "window.__METRICS__ = " + data_json + ";")
        .replace("/*{{DASHBOARD_JS}}*/", js)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html)

    n_cells = sum(len(r["cells"]) for r in data["runs"])
    print(f"Discovered {len(data['runs'])} run(s), {n_cells} config(s) "
          f"across platforms={data['platforms']} branches={data['branches']}.")
    print(f"Wrote {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"Open it with:  open '{OUT_PATH}'")
    return OUT_PATH


if __name__ == "__main__":
    build()
