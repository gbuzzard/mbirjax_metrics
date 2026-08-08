#!/usr/bin/env bash
# run_torch_regression.sh — fire-on-change nightly regression driver for MBIRTORCH.
# Sibling of run_regression.sh (the jax nightly), same two-phase shape, disjoint state:
#
# Phase 1 (bootstrap, runs from wherever scrontab/launchd invokes it): source the node preamble
#   (proxy + modules), update (or clone) the PERSISTENT metrics clone at $WORK_DIR/metrics, and
#   re-exec ITS copy of this script — so harness/engine/wrapper updates on the remote are always used.
# Phase 2 (work, runs from the persistent metrics clone): for each tracked branch whose remote head
#   moved since last measured (git ls-remote vs metrics state/<plat>/), make a SHALLOW single-branch
#   clone of the mbirtorch tip, install it into the dedicated env, run its test suite, run the torch
#   writer (torch_backend_writer.py — gate + records + dashboard files), record results + the
#   measured SHA into the metrics clone, and push (non-fatal).
#
# The two nightlies share NO mutable state: different WORK_DIR (hence lock + metrics clone),
# different conda env, different scrontab block, disjoint results/state paths (nightly_plan.md §8).
# Config: torch_regression.env.  Exits non-zero ONLY on a hard-gate regression (so the slurm mail
# from the mbirtorch-nightly job is a real alert); setup/transport hiccups are WARNs.
#
# Trial knobs (the §6 deployment seam; production leaves all three unset):
#   REG_TORCH_NO_PUSH=1   write results/state locally, skip commit/push (pre-schedule trial runs)
#   REG_TORCH_FORCE=1     treat every tracked branch as changed (re-measure an unmoved tip, e.g.
#                         the trial's second run exercising the vs-prior gate)
#   REG_TORCH_VENV=<dir>  use this existing venv instead of the dedicated conda env (the trial's
#                         TORCHPY-layered venv; see lib_torch_env.sh)
set -uo pipefail
# Keep an INTERACTIVE terminal open on a nonzero exit so the message stays visible; a tty-less run
# (launchd/scrontab/slurm) skips the pause.  Exit 1 is a REGRESSION (an alert — the run completed),
# exit >=2 is a harness/setup error.
if [ -t 0 ]; then
  trap '_ec=$?;
    if [ "$_ec" -eq 1 ]; then
      echo; echo ">>> $(basename "$0"): regression(s) DETECTED (exit 1) — an alert, not a failure.  Press Enter to close."; read -r _ || true
    elif [ "$_ec" -ne 0 ]; then
      echo; echo ">>> $(basename "$0") exited with status $_ec (harness/setup error) — press Enter to close."; read -r _ || true
    fi' EXIT
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/torch_regression.env"
log() { echo "[$(date '+%F %T')] $*"; }

[ "${ENABLED:-0}" = "1" ] || { log "ENABLED=0 — nothing to do."; exit 0; }

# Node preamble (cluster: `module load` conda/cuda + export HTTPS_PROXY; absent on the Mac).
# SOURCED in BOTH phases — conda's shell function does not survive the phase-1->2 exec.
if [ -n "${PREAMBLE_FILE:-}" ] && [ -f "$PREAMBLE_FILE" ]; then
  # shellcheck disable=SC1090
  source "$PREAMBLE_FILE"
  set -uo pipefail   # re-assert in case the preamble changed shell options
fi

# ── Phase 1: bootstrap — lock, update/clone the persistent metrics clone, re-exec ─────────────
if [ -z "${REG_TORCH_FRESH:-}" ]; then
  mkdir -p "$WORK_DIR"
  # Portable single-instance lock (macOS has no flock): mkdir is atomic everywhere.  This lock is
  # $WORK_DIR-scoped, so it serialises torch runs against each other and NEVER against the jax
  # nightly, whose lock lives in its own WORK_DIR.
  if ! mkdir "$WORK_DIR/.lock.d" 2>/dev/null; then
    log "another torch run holds the lock ($WORK_DIR/.lock.d) — exiting."; exit 0
  fi
  MC="$WORK_DIR/metrics"
  if [ -d "$MC/.git" ]; then
    log "updating metrics clone (fetch + rebase) -> $MC"
    if ! { git -C "$MC" fetch -q origin && git -C "$MC" pull -q --rebase --autostash; }; then
      log "metrics clone unusable — re-cloning fresh."
      rm -rf "$MC"
      git clone --quiet "$METRICS_URL" "$MC" || { log "FATAL: clone metrics failed."; rmdir "$WORK_DIR/.lock.d" 2>/dev/null; exit 2; }
    fi
  else
    log "cloning metrics -> $MC"
    git clone --quiet "$METRICS_URL" "$MC" || { log "FATAL: clone metrics failed."; rmdir "$WORK_DIR/.lock.d" 2>/dev/null; exit 2; }
  fi
  exec env REG_TORCH_FRESH=1 "$MC/tooling/regression/run_torch_regression.sh"
fi

# ── Phase 2: work — running from the persistent metrics clone ─────────────────────────────────
trap 'rm -rf "$WORK_DIR/.lock.d"' EXIT
METRICS_REPO="$(cd "$HERE/../.." && pwd)"        # = $WORK_DIR/metrics
HARNESS_DIR="$METRICS_REPO/tooling"

# Dedicated env (create if missing) + activate + harness deps; REG_TORCH_VENV substitutes an
# existing venv for trial runs (see lib_torch_env.sh).
# shellcheck disable=SC1091
source "$HARNESS_DIR/regression/lib_torch_env.sh"
reg_torch_activate_env || exit $?

# Platform DECLARATION (nvidia-smi signal, no torch import).  The writer VERIFIES this claim
# against torch.cuda.is_available() and hard-aborts on disagreement — a GPU night on which CUDA
# cannot initialise becomes a loud failure, never a run silently filed under the other key
# (nightly_plan.md §3(f); the 2026-07-21 jax lesson).
PLAT="$(reg_torch_plat)"
RES="$METRICS_REPO/results/$PLAT"; STATE="$METRICS_REPO/state/$PLAT"
mkdir -p "$RES" "$STATE"
log "platform=$PLAT env=${REG_TORCH_VENV:-$CONDA_ENV} metrics=$METRICS_REPO"

# Credential for unattended push (cluster), scoped to this repo only; same store the jax nightly
# uses.  GIT_TERMINAL_PROMPT=0 makes a missing credential fail fast instead of hanging the job.
export GIT_TERMINAL_PROMPT=0
if [ -n "${TOKEN_FILE:-}" ] && [ -f "$TOKEN_FILE" ]; then
  git -C "$METRICS_REPO" config credential.helper "store --file=$TOKEN_FILE"
fi

# ── Change detection via ls-remote (don't clone mbirtorch unless something moved) ──────────────
CHANGED_BR=()
for BR in "${TORCH_TRACKED_BRANCHES[@]}"; do
  SHA="$(git ls-remote "$MBIRTORCH_URL" "refs/heads/$BR" 2>/dev/null | awk '{print $1}')"
  [ -n "$SHA" ] || { log "skip $BR: not found on remote."; continue; }
  SLUG="${BR//\//_}"
  LAST="$(cat "$STATE/$SLUG" 2>/dev/null || true)"
  if [ "$SHA" = "$LAST" ] && [ "${REG_TORCH_FORCE:-0}" != "1" ]; then
    log "$BR @ ${SHA:0:8}: unchanged — skip."
  else
    [ -n "$LAST" ] && WAS="${LAST:0:8}" || WAS="none"
    if [ "$SHA" = "$LAST" ]; then
      log "$BR @ ${SHA:0:8}: unchanged but REG_TORCH_FORCE=1 — re-measuring."
    else
      log "$BR @ ${SHA:0:8}: CHANGED (was $WAS)."
    fi
    CHANGED_BR+=("$BR")
  fi
done
[ "${#CHANGED_BR[@]}" -gt 0 ] || { log "no tracked branch changed — done."; exit 0; }

DATE="$(date '+%Y%m%d')"
GATE_FAIL=0
TEST_FAIL=0
# Alert email: accumulate per-branch hard-gate items + test failures, sent ONCE at the end.  The
# exit code (and thus slurm's --mail-type=FAIL mail) tracks GATE_FAIL only, matching the jax
# convention: a test-only failure emails its detail WITHOUT flipping the exit code.
ALERT_BODY="$(mktemp)"
MAIL_TO="${REG_MAIL_TO:-buzzard@purdue.edu}"

for BR in "${CHANGED_BR[@]}"; do
  SLUG="${BR//\//_}"
  WT="$WORK_DIR/lib_$SLUG"; rm -rf "$WT"
  log "$BR: shallow-cloning the mbirtorch tip -> $WT"
  if ! git clone --quiet --depth 1 --branch "$BR" --single-branch "$MBIRTORCH_URL" "$WT"; then
    log "ERROR $BR: shallow clone failed — skip."; continue
  fi
  SHA="$(git -C "$WT" rev-parse HEAD)"   # the tip we actually got; recorded as state below

  log "$BR: installing mbirtorch [$TORCH_INSTALL_EXTRAS] into ${REG_TORCH_VENV:-$CONDA_ENV} ..."
  if ! reg_torch_install_lib "$WT" >"$WT/.install.log" 2>&1; then
    log "ERROR $BR: install of '$WT' failed (see $WT/.install.log) — skip."
    rm -rf "$WT"; continue
  fi

  # REG_TORCH_SMOKE = isolated plumbing test: a toy 1-cell sweep into a TEMP dir; skip tests /
  # commit / push / state below — never touches real data.
  if [ "${REG_TORCH_SMOKE:-0}" = "1" ]; then
    OUT="$(mktemp -d)"; log "$BR: REG_TORCH_SMOKE — toy output to $OUT (skipping tests / commit / push / state)."
  else
    OUT="$RES/$SLUG"
  fi
  mkdir -p "$OUT"

  # Tests FIRST, then the writer — the jax nightly's order.  NON-FATAL (logged + emailed, not
  # gated).  The wrapper owns this step (live tee, crash markers, alert lines); the writer's own
  # built-in suite run is switched off below via REG_TORCH_SKIP_TESTS.
  if [ "${RUN_TESTS:-0}" = "1" ] && [ "${REG_TORCH_SMOKE:-0}" != "1" ]; then
    TLOG="$OUT/tests_${PLAT}_${DATE}.txt"
    # xdist workers: 4 on GPU (concurrent CUDA inits abort in numbers), 8 on CPU.
    NPROC=$([ "$PLAT" = "gpu-torch" ] && echo 4 || echo 8)
    log "$BR: running the mbirtorch suite (xdist -n $NPROC) -> $(basename "$TLOG") ..."
    if [ -f "$WT/dev_scripts/run_tests.sh" ]; then
      # run_tests.sh resolves ../tests RELATIVE to dev_scripts/, so it must run from there.
      ( cd "$WT/dev_scripts" && PYTEST_NPROC="$NPROC" bash run_tests.sh ) 2>&1 | tee "$TLOG"
    else
      ( cd "$WT" && python -m pytest tests -ra -n "$NPROC" ) 2>&1 | tee "$TLOG"
    fi
    tests_rc="${PIPESTATUS[0]}"
    # A wholesale crash (xdist workers dying at import) can exit 0 with zero FAILED lines — scan
    # for crash markers so a run where NO tests ran cannot look green (the 2026-07-10 lesson).
    if grep -qaE "maximum crashed workers reached|Fatal Python error|INTERNALERROR|node down" "$TLOG" 2>/dev/null; then crashed=1; else crashed=0; fi
    if [ "$tests_rc" -eq 0 ] && [ "$crashed" -eq 0 ]; then
      log "$BR: tests done."
    elif [ "$crashed" -eq 1 ]; then
      TEST_FAIL=1
      log "$BR: TEST RUN CRASHED — xdist workers died; NO test results (emailed; see $(basename "$TLOG"))."
      { echo "### $BR @ ${SHA:0:8} — TEST RUN CRASHED (xdist workers died — no results)"
        grep -aE "maximum crashed workers reached|Fatal Python error|node down|INTERNALERROR" "$TLOG" | sort -u | head -6
        echo; } >>"$ALERT_BODY"
    else
      TEST_FAIL=1
      log "$BR: tests reported failures (non-fatal; emailed — see $(basename "$TLOG"))."
      { echo "### $BR @ ${SHA:0:8} — TEST FAILURES"
        grep -aE "^FAILED |[0-9]+ (failed|error)" "$TLOG" | tail -20
        echo; } >>"$ALERT_BODY"
    fi
  fi

  # The torch writer (gate + records + dashboard files).  It re-verifies the platform claim, pins
  # and asserts the device count on every row, and exits 1 on a hard-gate regression / >=2 on a
  # setup failure.  MBIRTORCH_MEMORY_CALIBRATION must be absent (the writer asserts it too).
  log "$BR: running the torch writer (output follows)..."
  GLOG="$(mktemp)"
  unset MBIRTORCH_MEMORY_CALIBRATION
  REG_TORCH_LIB_ROOT="$WT" REG_TORCH_OUT_DIR="$OUT" REG_TORCH_DATE="$DATE" REG_TORCH_GATE=1 \
       REG_TORCH_RUN_TAG="$BR" REG_TORCH_PLATFORM="$PLAT" REG_TORCH_SKIP_TESTS=1 \
       REG_TORCH_DEVICE_COUNTS="${TORCH_DEVICE_COUNTS:-1}" \
       REG_TORCH_MEM_GATE_WINDOW="${TORCH_MEM_GATE_WINDOW:-}" \
       python "$HARNESS_DIR/scaling_tests/torch_backend_writer.py" 2>&1 | tee "$GLOG"
  engine_rc="${PIPESTATUS[0]}"
  # The writer can also print an abort yet exit 0 in principle — catch CUDA/setup markers from the
  # output as the jax wrapper does, so a no-record night is loud, not silently green.
  if grep -qaE "produced no result|CUDA error|CUDA_ERROR|PLATFORM MISMATCH|DEVICE PIN FAILED" "$GLOG" 2>/dev/null; then engine_aborted=1; else engine_aborted=0; fi
  if [ "$engine_rc" -eq 0 ] && [ "$engine_aborted" -eq 0 ]; then
    log "$BR: writer ok."
  elif [ "$engine_rc" -eq 1 ] && [ "$engine_aborted" -eq 0 ]; then
    GATE_FAIL=1; log "$BR: GATE FAIL (regression) — see $OUT."
    { echo "### $BR @ ${SHA:0:8} — TORCH GATE FAIL (vs prior baseline)"
      grep -aE "GATE: FAIL|^ *HARD " "$GLOG" | head -25
      echo; } >>"$ALERT_BODY"
  else
    TEST_FAIL=1
    log "$BR: WRITER ABORTED (rc=$engine_rc) — no usable record; emailed."
    { echo "### $BR @ ${SHA:0:8} — TORCH WRITER ABORTED (rc=$engine_rc; no record -> no dashboard entry)"
      grep -aE "produced no result|CUDA error|CUDA_ERROR|PLATFORM MISMATCH|DEVICE PIN FAILED|Traceback" "$GLOG" | sort -u | head -6
      echo; } >>"$ALERT_BODY"
  fi
  rm -f "$GLOG"

  # Record the measured commit LAST, and only when the writer produced a record (rc 0 or 1 — a
  # gate FAIL still measured; an abort re-measures next run instead of being marked done).
  if [ "${REG_TORCH_SMOKE:-0}" != "1" ] && [ "$engine_aborted" != "1" ] && [ "$engine_rc" -le 1 ]; then
    echo "$SHA" >"$STATE/$SLUG"
  fi
  rm -rf "$WT"
done

# ── Publish to the metrics repo (conflict-safe; NON-FATAL) ────────────────────────────────────
# This wrapper stages ONLY its own platform's paths (results/<plat>/, state/<plat>/), which are
# disjoint from the jax nightly's — so concurrent pushes conflict only at the git level, where the
# rebase-retry below resolves them.  A failed push self-heals: state was not pushed, so the next
# run re-measures.  Skipped under REG_TORCH_SMOKE and under REG_TORCH_NO_PUSH (the trial seam).
if [ "${REG_TORCH_SMOKE:-0}" = "1" ]; then
  log "REG_TORCH_SMOKE — skipping commit/push."
elif [ "${REG_TORCH_NO_PUSH:-0}" = "1" ]; then
  log "REG_TORCH_NO_PUSH — results committed locally are NOT pushed (trial seam)."
else
git -C "$METRICS_REPO" add "results/$PLAT" "state/$PLAT" >/dev/null 2>&1 || true
git -C "$METRICS_REPO" diff --cached --name-only 2>/dev/null | while IFS= read -r f; do
  [ -f "$METRICS_REPO/$f" ] || continue
  mb=$(( $(wc -c <"$METRICS_REPO/$f") / 1048576 ))
  if [ "$mb" -gt "${MAX_PUSH_FILE_MB:-25}" ]; then
    git -C "$METRICS_REPO" reset -q -- "$f"
    log "WARN: not pushing oversized file ($mb MB > ${MAX_PUSH_FILE_MB:-25} MB): $f"
  fi
done
CHANGED_SUMMARY="$(IFS=,; echo "${CHANGED_BR[*]}")"
if git -C "$METRICS_REPO" commit -q -m "regression $PLAT $DATE [$CHANGED_SUMMARY]" >/dev/null 2>&1; then
  pushed=0
  for attempt in 1 2 3; do
    git -C "$METRICS_REPO" pull --rebase --autostash -q >/dev/null 2>&1 || true
    if git -C "$METRICS_REPO" push -q >/dev/null 2>&1; then pushed=1; break; fi
    log "push attempt $attempt failed (concurrent update?); rebasing + retrying."
  done
  [ "$pushed" = "1" ] && log "pushed results to metrics." \
    || log "WARN: push failed after 3 attempts; results not persisted (re-measures next run)."
else
  log "nothing new to commit."
fi
fi   # end publish guard

# ── Alert email: gate + test detail, addressed from the mbirtorch-nightly identity ────────────
if [ "$GATE_FAIL" != "0" ] || [ "$TEST_FAIL" != "0" ]; then
  SM="$(command -v sendmail || echo /usr/sbin/sendmail)"
  if [ -x "$SM" ]; then
    _brs="$(IFS=,; echo "${CHANGED_BR[*]}")"
    { printf 'Subject: [mbirtorch-nightly] %s regression: %s\nTo: %s\n\n' "$PLAT" "${_brs:-<none>}" "$MAIL_TO"
      echo "mbirtorch nightly ($PLAT) on $(hostname) at $(date '+%F %T')."
      echo "Branches measured this run: ${_brs:-<none>}"
      echo
      cat "$ALERT_BODY"
      [ -n "${SLURM_JOB_ID:-}" ] && echo "Full log: $WORK_DIR/nightly-${SLURM_JOB_ID}.log"
      echo "Records:  $RES/<branch>/  (record book records_${PLAT}.yaml)"
      echo
      echo "Note: a hard-gate regression AUTO-ADVANCES the baseline, so an alert fires ONCE per"
      echo "regressing change — review and revert if it is not an expected/accepted change.  Test"
      echo "failures are non-fatal (they do not change the exit code)."
    } | "$SM" -t && log "notify email sent to $MAIL_TO." || log "WARN: notify email send failed (non-fatal)."
  else
    log "WARN: no sendmail found — notify email skipped (slurm --mail-type=FAIL still covers gate fails)."
  fi
fi
rm -f "$ALERT_BODY"

[ "$GATE_FAIL" = "0" ] || { log "REGRESSION DETECTED — exit 1 (alert)."; exit 1; }
[ "$TEST_FAIL" = "0" ] && log "done — no regressions." \
  || log "done — no gate regression, but test failures/aborts occurred (emailed, non-fatal)."
exit 0
