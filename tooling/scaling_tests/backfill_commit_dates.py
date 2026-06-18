#!/usr/bin/env python3
"""One-time migration for the commit-time scheme.

For every existing result/golden YAML that lacks ``git_commit_date``, look up the committer date
of its recorded ``git_commit`` in a local mbirjax clone and insert it (as a minimal text edit, so
the rest of the file is byte-for-byte unchanged).  Then rename each run file to the commit-time
scheme ``regression_<plat>_<commitUTC>_<sha8>.yaml``.  Idempotent: files already carrying the date
and the new name are left untouched.  Golden/main_baseline files get the date but are not renamed.

The mbirjax clone is taken from ``MBIRJAX_REPO`` (default: the sibling ``../mbirjax`` of the metrics
repo).  Run it from the ``mbirjax`` conda env::

    python tooling/scaling_tests/backfill_commit_dates.py
"""
import datetime
import glob
import os
import re
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
METRICS = HERE.parents[1]                                  # tooling/scaling_tests -> tooling -> repo
MBIRJAX = Path(os.environ.get("MBIRJAX_REPO") or (METRICS.parent / "mbirjax"))


def _commit_date(sha):
    """Committer date (strict ISO-8601) of ``sha`` from the local mbirjax clone, or None."""
    if not sha:
        return None
    try:
        r = subprocess.run(["git", "-C", str(MBIRJAX), "show", "-s", "--format=%cI", sha],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if (r.returncode == 0 and r.stdout.strip()) else None
    except Exception:
        return None


def _utc_tag(iso, sha, fallback_date):
    """``<commitUTC>_<sha8>`` (matches performance_tracking._file_tag)."""
    sha8 = (sha or "")[:8]
    stamp = fallback_date
    if iso:
        try:
            stamp = datetime.datetime.fromisoformat(iso).astimezone(
                datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        except Exception:
            stamp = fallback_date
    return f"{stamp}_{sha8}" if sha8 else stamp


def _insert_date(path, iso):
    """Insert a top-level ``git_commit_date`` line right after ``git_commit`` (minimal text edit)."""
    text = Path(path).read_text()
    new = re.sub(r"(^git_commit:.*$)", rf"\1\ngit_commit_date: '{iso}'", text, count=1, flags=re.M)
    if new != text:
        Path(path).write_text(new)
        return True
    return False


def main():
    print(f"metrics = {METRICS}\nmbirjax = {MBIRJAX}")
    if not (MBIRJAX / ".git").exists():
        raise SystemExit(f"mbirjax repo not found at {MBIRJAX} (set MBIRJAX_REPO).")

    # Run files: add the date, then rename to the commit-time scheme.
    for path in sorted(glob.glob(str(METRICS / "results" / "*" / "*" / "regression_*.yaml"))):
        doc = yaml.safe_load(Path(path).read_text()) or {}
        iso = doc.get("git_commit_date")
        if not iso:
            iso = _commit_date(doc.get("git_commit"))
            if iso and _insert_date(path, iso):
                print(f"  + date  {os.path.basename(path)}")
        plat = doc.get("platform") or "?"
        tag = _utc_tag(iso, doc.get("git_commit"), str(doc.get("date") or ""))
        new = os.path.join(os.path.dirname(path), f"regression_{plat}_{tag}.yaml")
        if os.path.abspath(new) != os.path.abspath(path):
            os.rename(path, new)
            print(f"  rename  {os.path.basename(path)} -> {os.path.basename(new)}")

    # Golden / main_baseline: add the date only (fixed names).
    for path in sorted(glob.glob(str(METRICS / "golden" / "*.yaml"))):
        doc = yaml.safe_load(Path(path).read_text()) or {}
        if doc.get("git_commit_date"):
            continue
        iso = _commit_date(doc.get("git_commit"))
        if iso and _insert_date(path, iso):
            print(f"  + date  {os.path.basename(path)}")

    print("done.")


if __name__ == "__main__":
    main()
