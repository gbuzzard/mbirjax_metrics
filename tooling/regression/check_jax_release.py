#!/usr/bin/env python3
"""Warn (in the nightly log/email) when a jax NEWER than the last-reviewed version has shipped on PyPI.

Why: mbirjax excludes the known-bad jax 0.10.2 (an XLA codegen regression that ran GPU forward projection
3-9x slower; see mbirjax/.claude/lessons.md).  Otherwise mbirjax tracks the latest jax, so a future release
could re-introduce (or fix) such a regression silently.  This surfaces "a new jax is out" so it gets
re-tested with tooling/scaling_tests/measure_one_cell.py.  Workflow on an alert:
  - good  -> bump JAX_LAST_REVIEWED in action_scripts/run_configs.env to that version,
  - bad   -> add it to the `jax!=...` exclusion in mbirjax/pyproject.toml AND bump JAX_LAST_REVIEWED.
JAX_LAST_REVIEWED is the highest jax version we've ASSESSED (good or bad), so 0.10.2 belongs there even
though it's excluded -- the alert should fire only for versions past it.

Usage:  check_jax_release.py <last-reviewed-version>     e.g.  check_jax_release.py 0.10.2
Best-effort and NON-FATAL: any error (no network/proxy, bad arg) exits 0 silently so it never disturbs
the nightly.
"""
import json
import sys
import urllib.request

PYPI = "https://pypi.org/pypi/jax/json"


def _is_newer(latest, reviewed):
    try:
        from packaging.version import parse
        return parse(latest) > parse(reviewed)
    except Exception:
        import re
        tup = lambda v: tuple(int(x) for x in re.findall(r"\d+", v))
        try:
            return tup(latest) > tup(reviewed)
        except Exception:
            return latest != reviewed   # last resort: any difference


def _pypi_latest():
    try:
        with urllib.request.urlopen(PYPI, timeout=15) as r:
            return json.load(r)["info"]["version"]
    except Exception:
        return None   # offline / proxy hiccup


def main(argv):
    # `--print-latest`: emit just the latest PyPI jax version (nothing on failure) for the dependency
    # canary's fingerprint (run_regression.sh compares it to state/jax_seen).
    if len(argv) > 1 and argv[1] == "--print-latest":
        v = _pypi_latest()
        if v:
            print(v)
        return 0
    reviewed = (argv[1].strip() if len(argv) > 1 else "")
    if not reviewed:
        return 0
    latest = _pypi_latest()
    if latest is None:
        return 0   # offline / proxy hiccup -> stay silent, never fail the nightly
    if _is_newer(latest, reviewed):
        print(f"[jax-watch] NEW jax on PyPI: {latest}  (last reviewed: {reviewed}).  Re-test it with "
              f"tooling/scaling_tests/measure_one_cell.py; if good, bump JAX_LAST_REVIEWED in "
              f"run_configs.env; if it regresses (cf. the 0.10.2 forward slowdown), add it to the "
              f"jax!=... exclusion in mbirjax/pyproject.toml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
