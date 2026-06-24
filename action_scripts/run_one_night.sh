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
# On a SLURM cluster, add --sbatch to SUBMIT this pass as a batch job on a GPU node (resources from
# run_configs.env's SLURM_* knobs) instead of running it in this interactive session.
#
# Safe whether run with `bash`/`./` or `source`d: the work runs in a subshell so `set -e`/traps stay
# isolated; a nonzero exit pauses (keeping the terminal open) instead of closing it.
#
# Exit-code meaning (faithful to the scheduler): 0 = clean; 1 = a hard-gate perf REGRESSION was
# detected (the run still completed and pushed — this is an ALERT, not a script failure); >=2 =
# the harness itself failed (clone/install/conda/transport).  The trap below words these distinctly
# so a regression doesn't read as a crash.

if (return 0 2>/dev/null); then _sourced=1; else _sourced=0; fi

# --sbatch (cluster): resubmit this pass as a SLURM batch job (minus the flag) and exit, so the nightly
# runs on a GPU compute node instead of an interactive session.  See tooling/regression/sbatch_submit.sh.
case " $* " in *" --sbatch "*)
  _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; _REPO="$(cd "$_HERE/.." && pwd)"
  # shellcheck disable=SC1091
  source "$_REPO/tooling/regression/regression.env"
  # shellcheck disable=SC1091
  source "$_REPO/tooling/regression/sbatch_submit.sh"
  _ARGS=(); for _a in "$@"; do [ "$_a" = "--sbatch" ] || _ARGS+=("$_a"); done
  submit_sbatch "mbirjax-night" bash "$_HERE/run_one_night.sh" "${_ARGS[@]}"
  _rc=$?
  if [ "$_sourced" -eq 1 ]; then return "$_rc"; else exit "$_rc"; fi
  ;;
esac

(
  trap 'rc=$?;
        if [ "$rc" -eq 1 ]; then
          printf "\nrun_one_night.sh: completed — hard-gate regression(s) DETECTED (exit 1).\n" >&2
          printf "This is an ALERT, not a failure: the run finished and results were recorded + pushed.\n" >&2
          [ -t 0 ] && read -r -p "Press Enter to close... " _ </dev/tty || true
        elif [ "$rc" -ne 0 ]; then
          printf "\nrun_one_night.sh FAILED (exit %s) — harness/setup error.\n" "$rc" >&2
          [ -t 0 ] && read -r -p "Press Enter to close... " _ </dev/tty || true
        fi' EXIT
  set -euo pipefail
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # Detach the harness's stdin (</dev/null) so ITS own interactive trap (which also pauses on a
  # nonzero exit) never installs — this wrapper owns the single user-facing message + pause above.
  "$HERE/../tooling/regression/run_regression.sh" "$@" </dev/null
)
_rc=$?
if [ "$_sourced" -eq 1 ]; then return "$_rc"; else exit "$_rc"; fi
