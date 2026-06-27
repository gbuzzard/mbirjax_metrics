"""
experiments/profiling/profile_diff.py
────────────────────────────────────────
Before/after a redesign: diff two profile_<plat>_*.yaml runs region by region.

This is the headline use of the region schema — "did the redesign move the bottleneck?".  For each
cell present in both runs it shows the wall-time change and, per named_scope region, how its share
(pct) and wall-attributed ms moved, plus regions that appeared/vanished (so adding/removing a
named_scope is handled).

GUARD: profiling must compare like with like — the diff REFUSES (loud warning) when the two runs
used a different jax/jaxlib, because a cross-version comparison is meaningless (we learned this the
hard way: the 0.10.2 regression).  Same reason both runs must be the same platform.

No CLI args.  Edit the CONFIG block: pick two explicit files, or leave them None to auto-diff the
two most recent runs for PLATFORM (the common "what changed since my last run" case).
"""
import os
import glob

from ruamel.yaml import YAML

# ── CONFIG (edit here) ────────────────────────────────────────────────────────
PLATFORM = "cpu"        # which platform's runs to auto-diff: "cpu" or "gpu"
BEFORE = None           # explicit path to the BEFORE run YAML, or None = 2nd-most-recent for PLATFORM
AFTER = None            # explicit path to the AFTER  run YAML, or None =   most-recent for PLATFORM

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "results")
_yaml = YAML(typ="safe")


def _load(path):
    with open(path) as f:
        return _yaml.load(f)


def _pick():
    if BEFORE and AFTER:
        return BEFORE, AFTER
    files = sorted(glob.glob(os.path.join(_RESULTS, f"profile_{PLATFORM}_*.yaml")))
    if len(files) < 2:
        raise SystemExit(f"Need >=2 profile_{PLATFORM}_*.yaml runs in {_RESULTS} (found {len(files)}). "
                         f"Run profile_measure.py at two commits, or set BEFORE/AFTER.")
    return files[-2], files[-1]


def _delta(a, b):
    """(b - a, percent change) — percent is None when a is 0."""
    d = b - a
    return d, (100.0 * d / a if a else None)


def _fmt_pct_change(pct):
    return "  n/a" if pct is None else f"{pct:+5.1f}%"


def main():
    bpath, apath = _pick()
    B, A = _load(bpath), _load(apath)
    br, ar = B.get("run", {}), A.get("run", {})
    benv, aenv = br.get("env", {}), ar.get("env", {})

    print("=" * 90)
    print("  PROFILE DIFF   (BEFORE -> AFTER)")
    print(f"  BEFORE: {os.path.basename(bpath)}   mbirjax {(br.get('mbirjax_commit') or '?')[:8]}  jax {benv.get('jax')}")
    print(f"  AFTER : {os.path.basename(apath)}   mbirjax {(ar.get('mbirjax_commit') or '?')[:8]}  jax {aenv.get('jax')}")
    print("=" * 90)
    if br.get("platform") != ar.get("platform"):
        print(f"  !! REFUSING: platforms differ ({br.get('platform')} vs {ar.get('platform')}); not comparable.")
        return
    if benv.get("jax") != aenv.get("jax") or benv.get("jaxlib") != aenv.get("jaxlib"):
        print(f"  !! WARNING: jax/jaxlib differ ({benv.get('jax')}/{benv.get('jaxlib')} vs "
              f"{aenv.get('jax')}/{aenv.get('jaxlib')}) — a cross-version diff is NOT meaningful. Stopping.")
        return

    bc, ac = B.get("cells", {}) or {}, A.get("cells", {}) or {}
    for key in sorted(set(bc) | set(ac)):
        cb, ca = bc.get(key), ac.get(key)
        if cb is None:
            print(f"\n### {key}   [NEW cell — only in AFTER]"); continue
        if ca is None:
            print(f"\n### {key}   [dropped — only in BEFORE]"); continue
        wd, wp = _delta(cb["wall_ms"], ca["wall_ms"])
        print(f"\n### {key}")
        print(f"  wall_ms: {cb['wall_ms']:.1f} -> {ca['wall_ms']:.1f}   ({wd:+.1f} ms, {_fmt_pct_change(wp)})")
        rb, ra = cb.get("regions", {}) or {}, ca.get("regions", {}) or {}
        print(f"  {'region':<34}{'pct (A->B)':>16}{'Δpp':>8}{'ms (A->B)':>20}{'Δms':>10}")
        print("  " + "-" * 86)
        order = sorted(set(rb) | set(ra), key=lambda r: -max(rb.get(r, {}).get("pct", 0), ra.get(r, {}).get("pct", 0)))
        for r in order:
            vb, va = rb.get(r), ra.get(r)
            if vb is None:
                print(f"  {r:<34}{'— -> %.1f' % va['pct']:>16}{'+%.1f' % va['pct']:>8}"
                      f"{'— -> %.1f' % va['ms']:>20}{'+%.1f' % va['ms']:>10}   [added]")
            elif va is None:
                print(f"  {r:<34}{'%.1f -> —' % vb['pct']:>16}{'-%.1f' % vb['pct']:>8}"
                      f"{'%.1f -> —' % vb['ms']:>20}{'-%.1f' % vb['ms']:>10}   [removed]")
            else:
                dpp = va["pct"] - vb["pct"]
                dms = va["ms"] - vb["ms"]
                print(f"  {r:<34}{'%.1f -> %.1f' % (vb['pct'], va['pct']):>16}{dpp:>+8.1f}"
                      f"{'%.1f -> %.1f' % (vb['ms'], va['ms']):>20}{dms:>+10.1f}")


if __name__ == "__main__":
    main()
