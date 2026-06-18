#!/usr/bin/env bash
# Install + start the scheduled nightly regression (forwards to tooling/regression/enable_nightly.sh).
# Platform-aware: macOS launchd works now; the cluster (scrontab + nightly_regression.slurm) is
# pending the slurm script.  Safe whether run with bash/./ or sourced; keeps the terminal open on a
# nonzero exit.
if (return 0 2>/dev/null); then _sourced=1; else _sourced=0; fi
(
  trap 'rc=$?; if [ "$rc" -ne 0 ]; then
          printf "\nenable_nightly.sh failed (exit %s).\n" "$rc" >&2
          [ -t 0 ] && read -r -p "Press Enter to close... " _ </dev/tty || true
        fi' EXIT
  set -euo pipefail
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  "$HERE/../tooling/regression/enable_nightly.sh" "$@"
)
_rc=$?
if [ "$_sourced" -eq 1 ]; then return "$_rc"; else exit "$_rc"; fi
