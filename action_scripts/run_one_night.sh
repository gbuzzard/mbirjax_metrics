#!/usr/bin/env bash
# Run ONE nightly regression pass right now, by hand.
#
# Use this to verify the pipeline (and start collecting data) before enabling the scheduled nightly.
# It forwards to the real harness, `tooling/regression/run_regression.sh` — the exact code the
# scheduler will run — so this is a faithful single pass, not a stand-in: for each tracked branch
# whose remote tip moved since it was last measured, it clones the tip, runs the tests + the perf
# engine, writes results into the harness's persistent metrics clone, and pushes to GitHub.
#
# The harness manages its own conda env (CONDA_ENV in regression.env, default `mbirjax_regression`,
# auto-created) and, on the cluster, sources PREAMBLE_FILE — so this wrapper does NOT activate a env.
# To inspect afterwards: pull the pushed results into this working repo, then run
# `action_scripts/build_dashboard.sh`.  (Set REG_SMOKE=1 for a fast 1-cell plumbing check.)
#
# Safe whether run with `bash`/`./` or `source`d: the work runs in a subshell so `set -e`/traps stay
# isolated; a nonzero exit pauses (keeping the terminal open) instead of closing it.

if (return 0 2>/dev/null); then _sourced=1; else _sourced=0; fi

(
  trap 'rc=$?; if [ "$rc" -ne 0 ]; then
          printf "\nrun_one_night.sh failed (exit %s).\n" "$rc" >&2
          [ -t 0 ] && read -r -p "Press Enter to close... " _ </dev/tty || true
        fi' EXIT
  set -euo pipefail
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  "$HERE/../tooling/regression/run_regression.sh" "$@"
)
_rc=$?
if [ "$_sourced" -eq 1 ]; then return "$_rc"; else exit "$_rc"; fi
