#!/usr/bin/env bash
# run_regression.sh — fire-on-change nightly regression driver (FRESH-CLONE model).
#
# Phase 1 (bootstrap, runs from wherever cron/launchd/scrontab invokes it): source the node preamble
#   (proxy + modules), update (or clone) the PERSISTENT metrics clone at $WORK_DIR/metrics, and re-exec
#   ITS copy of this script — so harness/engine/wrapper updates on the remote are always used.
# Phase 2 (work, runs from the persistent metrics clone): for each tracked branch whose remote head moved
#   since last measured (git ls-remote vs metrics state/), make a SHALLOW single-branch clone of the
#   library tip, pip install -e (+platform extras), run that branch's tests + the perf engine, record
#   results + the measured SHA into the metrics clone, and push (non-fatal).
#
# No per-node paths are baked in — only URLs + $WORK_DIR (under $HOME or scratch).  Config: regression.env.
# Exits non-zero ONLY on a hard-gate perf regression (so the cron/slurm mail is a real alert); setup/
# transport hiccups are WARNs.
set -uo pipefail
# Keep an INTERACTIVE terminal open on a nonzero exit so the message stays visible (some terminals
# close the window when a command exits nonzero).  A tty-less run (cron/launchd/scrontab/slurm) has
# no stdin tty, so it skips the pause and exits with the real code — the alert path is preserved.
# Word the two nonzero cases distinctly: exit 1 is a perf REGRESSION (an alert — the run completed),
# exit >=2 is a harness/setup error.  (When invoked via run_one_night.sh, stdin is detached so this
# trap does not install and that wrapper owns the single message + pause.)
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
source "$HERE/regression.env"
log() { echo "[$(date '+%F %T')] $*"; }

[ "${ENABLED:-0}" = "1" ] || { log "ENABLED=0 — nothing to do."; exit 0; }

# Node preamble (cluster: `module load` conda/cuda + export HTTPS_PROXY; empty on the Mac).  SOURCED
# in BOTH phases: phase 1 needs the proxy to git-clone from github, phase 2 needs conda/cuda — and
# conda's shell function does NOT survive the phase-1->2 exec, so phase 2 must re-source, not inherit.
# It is SOURCED, so PREAMBLE_FILE must be source-safe: module loads + exports only, NO `exit`.
if [ -n "${PREAMBLE_FILE:-}" ] && [ -f "$PREAMBLE_FILE" ]; then
  # shellcheck disable=SC1090
  source "$PREAMBLE_FILE"
  set -uo pipefail   # re-assert in case the preamble changed shell options (e.g. turned on set -e)
fi

# ── Phase 1: bootstrap — lock, update/clone the persistent metrics clone, re-exec ─────────────
if [ -z "${REG_FRESH:-}" ]; then
  mkdir -p "$WORK_DIR"
  # Portable single-instance lock (macOS has no flock): mkdir is atomic everywhere.  Held as a dir
  # on disk across the exec; phase 2 sets the EXIT trap that removes it.
  if ! mkdir "$WORK_DIR/.lock.d" 2>/dev/null; then
    log "another run holds the lock ($WORK_DIR/.lock.d) — exiting."; exit 0
  fi
  # PERSISTENT metrics clone (NOT throwaway): reuse it across runs so a FAILED push never loses a
  # run's results — they stay committed locally and the next run's rebase-retry pushes them (and you
  # can `git -C "$WORK_DIR/metrics" push` by hand).  Reuse = fetch + rebase (gets remote changes while
  # keeping any unpushed local commits, public repo so no creds needed for read); re-clone only if the
  # existing clone is missing or unusable.  NOTE: this entrypoint must be a SEPARATE checkout from
  # $WORK_DIR/metrics (the cron/standing wrapper), so updating it here never modifies the running script.
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
  # Re-exec the updated copy (picks up remote wrapper/engine changes).  The env (proxy/modules from the
  # preamble, the lock dir on disk) carries through exec.
  exec env REG_FRESH=1 "$MC/tooling/regression/run_regression.sh"
fi

# ── Phase 2: work — running from the FRESH metrics clone ──────────────────────────────────────
trap 'rm -rf "$WORK_DIR/.lock.d"' EXIT
METRICS_REPO="$(cd "$HERE/../.." && pwd)"        # = $WORK_DIR/metrics
HARNESS_DIR="$METRICS_REPO/tooling"

# Dedicated env (create if missing) + activate + harness deps — the SHARED mechanism, so a manual
# add_run backfill is produced by the same pipeline (see lib_env.sh).  conda must be reachable here
# (Mac: your shell PATH; cluster: from PREAMBLE_FILE's `module load conda`); the `conda activate` shell
# function is re-sourced inside, since it does not survive the phase-1->2 exec.
# shellcheck disable=SC1091
source "$HARNESS_DIR/regression/lib_env.sh"
reg_activate_env || exit $?

# Platform + pip extras (shared with add_run; no jax import in the wrapper — just the GPU signal).
read -r PLAT EXTRAS <<<"$(reg_plat_extras)"
RES="$METRICS_REPO/results/$PLAT"; STATE="$METRICS_REPO/state/$PLAT"
mkdir -p "$RES" "$STATE"
log "platform=$PLAT extras=[$EXTRAS] env=$CONDA_ENV metrics=$METRICS_REPO"

# Credential for unattended push (cluster), scoped to this repo only.  TOKEN_FILE must be a git
# credential-STORE file (chmod 600), i.e. ONE line:  https://<user>:<PAT>@github.com  (not the bare
# token).  Never prompt interactively: GIT_TERMINAL_PROMPT=0 makes a missing/invalid credential fail
# fast (a WARN on push) instead of hanging an unattended job on a password prompt.
export GIT_TERMINAL_PROMPT=0
if [ -n "${TOKEN_FILE:-}" ] && [ -f "$TOKEN_FILE" ]; then
  git -C "$METRICS_REPO" config credential.helper "store --file=$TOKEN_FILE"
fi

# jax-release watch (best-effort, NON-FATAL, runs every night even on no-change): emit a one-line WARN
# into the log/email when a jax newer than JAX_LAST_REVIEWED ships, so the 0.10.2-style forward-perf
# regression gets re-tested (measure_one_cell.py) instead of silently riding in on the next clean
# install.  Placed BEFORE the no-change exit below so it fires regardless of branch activity.
if [ -n "${JAX_LAST_REVIEWED:-}" ] && [ -f "$HERE/check_jax_release.py" ]; then
  python "$HERE/check_jax_release.py" "$JAX_LAST_REVIEWED" 2>/dev/null || true
fi

# ── Change detection via ls-remote (don't clone mbirjax unless something moved) ────────────────
CHANGED_BR=(); CHANGED_SHA=()
for BR in "${TRACKED_BRANCHES[@]}"; do
  SHA="$(git ls-remote "$MBIRJAX_URL" "refs/heads/$BR" 2>/dev/null | awk '{print $1}')"
  [ -n "$SHA" ] || { log "skip $BR: not found on remote."; continue; }
  SLUG="${BR//\//_}"
  LAST="$(cat "$STATE/$SLUG" 2>/dev/null || true)"
  if [ "$SHA" = "$LAST" ]; then
    log "$BR @ ${SHA:0:8}: unchanged — skip."
  else
    [ -n "$LAST" ] && WAS="${LAST:0:8}" || WAS="none"
    log "$BR @ ${SHA:0:8}: CHANGED (was $WAS)."
    CHANGED_BR+=("$BR"); CHANGED_SHA+=("$SHA")
  fi
done

# ── Dependency canary: a NEW jax counts as a change (dependency_canary_plan.md) ────────────────
# Guarded by DEP_CANARY_ENABLED (default 0 -> this whole feature is inert; the nightly is unchanged).
# When PyPI's latest jax differs from state/jax_seen, bump the dep-generation counter and ensure the
# canary branch is measured (even if its tip didn't move) so the new jax is re-measured + attributed.
NEW_JAX=0; DEP_GEN=""; PYPI_JAX=""
if [ "${DEP_CANARY_ENABLED:-0}" = "1" ] && [ -f "$HERE/check_jax_release.py" ]; then
  CANARY="${DEP_CANARY_BRANCH:-main}"
  PYPI_JAX="$(python "$HERE/check_jax_release.py" --print-latest 2>/dev/null || true)"
  SEEN="$(cat "$STATE/jax_seen" 2>/dev/null || true)"
  if [ -n "$PYPI_JAX" ] && [ "$PYPI_JAX" != "$SEEN" ]; then
    NEW_JAX=1
    DEP_GEN="$(( $(cat "$STATE/depgen" 2>/dev/null || echo 0) + 1 ))"
    log "dep-canary: jax $PYPI_JAX on PyPI (was ${SEEN:-none}) -> dep-gen $DEP_GEN; upgrade + re-measure $CANARY."
    _in=0; for _b in "${CHANGED_BR[@]}"; do [ "$_b" = "$CANARY" ] && _in=1; done
    if [ "$_in" = "0" ]; then
      _csha="$(git ls-remote "$MBIRJAX_URL" "refs/heads/$CANARY" 2>/dev/null | awk '{print $1}')"
      [ -n "$_csha" ] && { CHANGED_BR+=("$CANARY"); CHANGED_SHA+=("$_csha"); \
        log "dep-canary: added $CANARY @ ${_csha:0:8} (tip unchanged) for the jax re-measure."; }
    fi
  fi
fi

[ "${#CHANGED_BR[@]}" -gt 0 ] || { log "no tracked branch changed — done."; exit 0; }

# Per changed branch: a SHALLOW, SINGLE-BRANCH clone of just the branch tip (no history, no other
# branches) straight into the work dir — small + fast, and (unlike a pip-install-from-git, which
# clones internally anyway) it still carries tests/ + dev_scripts/ for the test step and a .git for
# provenance.  The big experiments/ tree is gitignored, so it is never cloned.
DATE="$(date '+%Y%m%d')"
GATE_FAIL=0

# dep-canary: force jax/jaxlib to latest in the shared env BEFORE the branch loop, so every branch this
# night measures the new jax (the per-branch editable install re-resolves + pulls an excluded version
# back down — §4).  Non-fatal; skipped in smoke.
if [ "$NEW_JAX" = "1" ] && [ "${REG_SMOKE:-0}" != "1" ]; then
  log "dep-canary: upgrading jax/jaxlib to latest in $CONDA_ENV ..."
  reg_upgrade_jax "$EXTRAS" >/dev/null 2>&1 || log "dep-canary: WARN jax upgrade failed — using current jax."
fi

for i in "${!CHANGED_BR[@]}"; do
  BR="${CHANGED_BR[$i]}"; SLUG="${BR//\//_}"
  WT="$WORK_DIR/lib_$SLUG"; rm -rf "$WT"
  log "$BR: shallow-cloning the library tip -> $WT"
  if ! git clone --quiet --depth 1 --branch "$BR" --single-branch "$MBIRJAX_URL" "$WT"; then
    log "ERROR $BR: shallow clone failed — skip."; continue
  fi
  SHA="$(git -C "$WT" rev-parse HEAD)"   # the tip we actually got; recorded as state below

  log "$BR: installing library [$EXTRAS] into $CONDA_ENV (first time pulls jax — can be slow)..."
  if ! reg_install_lib "$WT" "$EXTRAS" >"$WT/.install.log" 2>&1; then
    log "ERROR $BR: pip install -e '$WT[$EXTRAS]' failed (see $WT/.install.log) — skip."
    rm -rf "$WT"; continue
  fi

  # REG_SMOKE = isolated plumbing test: exercise the whole flow with a TOY 1-cell engine into a TEMP
  # dir, and skip tests / commit / push / state below — so it never touches real data.  Use it to
  # verify the wrapper (persistence, clone, install, engine) in ~1-2 min without a full measurement.
  if [ "${REG_SMOKE:-0}" = "1" ]; then
    OUT="$(mktemp -d)"; log "$BR: REG_SMOKE — toy output to $OUT (skipping tests / commit / push / state)."
  else
    OUT="$RES/$SLUG"
  fi
  mkdir -p "$OUT"

  # Tests: reuse the branch's OWN runner (your -n 10 tuning + conftest knobs); NON-FATAL (logged,
  # not gated — per-branch test diffing is a later increment; the perf engine is the alert path).
  # `tee` shows progress live (interactive) AND records to tests_*.txt; on an unattended run the
  # stdout copy just lands in the cron/slurm log.
  if [ "${RUN_TESTS:-0}" = "1" ] && [ "${REG_SMOKE:-0}" != "1" ]; then
    TLOG="$OUT/tests_${PLAT}_${DATE}.txt"
    log "$BR: running tests (MBIRJAX_NUM_CPU_DEVICES=$TEST_CPU_DEVICES) -> $(basename "$TLOG") ..."
    if [ -f "$WT/dev_scripts/run_tests.sh" ]; then
      # run_tests.sh uses a path RELATIVE to dev_scripts/ (`python -m pytest -n 10 ../tests`), so it
      # MUST be invoked from there or it collects 0 tests (../tests would resolve outside the clone).
      ( cd "$WT/dev_scripts" && MBIRJAX_NUM_CPU_DEVICES="$TEST_CPU_DEVICES" bash run_tests.sh ) 2>&1 | tee "$TLOG"
    else
      ( cd "$WT" && MBIRJAX_NUM_CPU_DEVICES="$TEST_CPU_DEVICES" python -m pytest tests -q -n 10 ) 2>&1 | tee "$TLOG"
    fi
    [ "${PIPESTATUS[0]}" -eq 0 ] || log "$BR: tests reported failures (non-fatal; see $(basename "$TLOG"))."
    log "$BR: tests done."
  fi

  # Perf engine (fixed harness; lib_root=$WT selects the library + provenance).  The gate compares
  # each run against this branch's own prior run; cross-branch + best-ever drift are shown on the
  # dashboard.  run_nightly.py prints to stdout, so the engine output shows live on a manual run
  # (and lands in the cron/slurm log unattended).
  # dep-canary provenance: the canary branch's run this night is the jax re-measure at dep-gen NNNN.
  # Empty for every other run -> run_nightly.py ignores them -> dep_gen 0 / run_reason "commit".
  DGEN=""; RREASON=""
  if [ "$NEW_JAX" = "1" ] && [ "$BR" = "${DEP_CANARY_BRANCH:-main}" ]; then DGEN="$DEP_GEN"; RREASON="jax-step"; fi
  log "$BR: running perf engine (output follows)..."
  if REG_LIB_ROOT="$WT" REG_OUT_DIR="$OUT" REG_DATE="$DATE" REG_GATE=1 REG_RUN_TAG="$BR" \
       REG_DEP_GEN="$DGEN" REG_RUN_REASON="$RREASON" \
       python "$HARNESS_DIR/scaling_tests/run_nightly.py"; then
    log "$BR: engine ok."
  else
    GATE_FAIL=1; log "$BR: GATE FAIL (perf regression) — see $OUT."
  fi

  # Record the measured commit LAST (a crash mid-run re-measures next time).  Skipped in smoke so a
  # test run never marks the branch as measured.
  [ "${REG_SMOKE:-0}" = "1" ] || echo "$SHA" >"$STATE/$SLUG"
  rm -rf "$WT"                  # drop the throwaway library clone
done

# dep-canary: record the PyPI jax we acted on + the new dep-gen LAST (a crash mid-run re-fires next
# time, like the branch state above).  jax_seen is the PyPI-latest we've SEEN, so an excluded release
# won't re-fire nightly; the actual installed jax lives in each run's toolchain block.
if [ "$NEW_JAX" = "1" ] && [ "${REG_SMOKE:-0}" != "1" ]; then
  echo "$PYPI_JAX" >"$STATE/jax_seen"
  echo "$DEP_GEN"  >"$STATE/depgen"
fi

# ── Publish to the metrics repo (conflict-safe; NON-FATAL) ────────────────────────────────────
# CPU (Mac) and GPU (cluster) write DISJOINT paths (results/<plat>/, state/<plat>/, *_<plat>.yaml),
# so concurrent runs never conflict on CONTENT — only at the git level (a non-fast-forward if the
# other platform pushed between our clone and our push).  So: rebase onto the latest (always clean,
# the paths don't overlap) and retry.  If the push ultimately fails (auth/network), this run's
# results+state simply aren't persisted — which SELF-HEALS: next run sees no new state and re-measures.
# Skipped under REG_SMOKE — the isolated plumbing test must not touch the real metrics repo.
if [ "${REG_SMOKE:-0}" = "1" ]; then
  log "REG_SMOKE — skipping commit/push."
else
git -C "$METRICS_REPO" add results state >/dev/null 2>&1 || true
# Size cap (backstop): unstage any staged file larger than MAX_PUSH_FILE_MB so a stray large
# artifact can never be pushed.  Normal outputs (YAML/.txt/.npy) are well under this.
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
fi   # end REG_SMOKE publish guard

[ "$GATE_FAIL" = "0" ] || { log "REGRESSION DETECTED — exit 1 (alert)."; exit 1; }
log "done — no regressions."
exit 0
