"""
experiments/profiling/profile_to_table.py
────────────────────────────────────────────
Render a profile_<plat>_*.yaml run as a browsable, diff-able nested YAML — the regression_to_table.py
analog for the region schema.  It nests geometry -> op -> size -> n_devices, and under each cell lists
the named_scope regions sorted by share as aligned ``pct%   ms`` strings.

Output is itself YAML, so it parses cleanly AND PyCharm's structure breadcrumb accumulates the
``GEOMETRY .. / OP .. / size .. / n=..`` path as you scroll (same trick as regression_to_table: the
label is folded into the key — ``GEOMETRY cone:`` not ``GEOMETRY: cone`` — because a YAML key can't
carry both a scalar value and child keys).

No CLI args: edit CONFIG (PLATFORM + an explicit RUN path, or None = most-recent for PLATFORM).
Writes ``<input>_table.yaml`` next to the run and echoes it.
"""
import os
import sys
import glob

from ruamel.yaml import YAML

# ── CONFIG (edit here) ────────────────────────────────────────────────────────
PLATFORM = "cpu"        # which platform's run to render: "cpu" or "gpu"
RUN = None              # explicit profile_*.yaml path, or None = most-recent for PLATFORM

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "results")
GEOM_ORDER = ["parallel", "cone", "translation", "multiaxis_parallel", "denoiser"]
OP_ORDER = ["direct_filter", "forward", "back", "vcd_nonconst", "denoise"]
_yaml = YAML()
_yaml.default_flow_style = False


def _pick():
    if RUN:
        return RUN
    files = sorted(glob.glob(os.path.join(_RESULTS, f"profile_{PLATFORM}_*.yaml")))
    if not files:
        raise SystemExit(f"No profile_{PLATFORM}_*.yaml in {_RESULTS}. Run profile_measure.py first.")
    return files[-1]


def _order(value, order):
    return (order.index(value) if value in order else len(order), str(value))


def _vol(size):
    try:
        v = 1
        for x in size.split("x"):
            v *= int(x)
        return v
    except Exception:   # noqa: BLE001
        return 0


def main():
    path = _pick()
    with open(path) as f:
        doc = _yaml.load(f)
    run, cells = doc.get("run", {}) or {}, doc.get("cells", {}) or {}

    out = {}
    out["run"] = {k: run[k] for k in ("mbirjax_commit", "mbirjax_commit_date", "collected", "platform")
                  if k in run}
    if run.get("env"):
        out["run"]["env"] = run["env"]

    # group cells (geom|op|size|ndev) into geom -> op -> size -> ndev
    tree = {}
    for key, cell in cells.items():
        parts = key.split("|")
        if len(parts) != 4:
            continue
        geom, op, size, ndev = parts
        tree.setdefault(geom, {}).setdefault(op, {}).setdefault(size, {})[ndev] = cell

    for geom in sorted(tree, key=lambda g: _order(g, GEOM_ORDER)):
        gnode = out[f"GEOMETRY {geom}"] = {}
        for op in sorted(tree[geom], key=lambda o: _order(o, OP_ORDER)):
            onode = gnode[f"OP {op}"] = {}
            for size in sorted(tree[geom][op], key=_vol):
                snode = onode[f"size {size}"] = {}
                for ndev in sorted(tree[geom][op][size], key=lambda n: int(n)):
                    cell = tree[geom][op][size][ndev]
                    regions = cell.get("regions", {}) or {}
                    items = sorted(regions.items(), key=lambda kv: -(kv[1] or {}).get("pct", 0))
                    rnode = {r: f"{v.get('pct', 0):5.1f}%   {v.get('ms', 0):10.1f} ms" for r, v in items}
                    node = {"wall_ms": cell.get("wall_ms"), "regions": rnode}
                    # Flag any base-name collisions: the assigned region's ms is uncertain by up to this
                    # much because the join can't split a fusion that spans >1 named_scope (GPU-typical).
                    cols = cell.get("collisions") or []
                    if cols:
                        node["collisions"] = [
                            f"{c.get('base')}: up to {c.get('ms', 0):.1f} ms uncertain"
                            f"  (-> {c.get('assigned')}; spans {dict(c.get('scopes', {}))})"
                            for c in cols]
                    snode[f"n={ndev}"] = node

    outpath = (path[:-5] if path.endswith(".yaml") else path) + "_table.yaml"
    with open(outpath, "w") as f:
        _yaml.dump(out, f)
    print(f"wrote {outpath}\n")
    _yaml.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
