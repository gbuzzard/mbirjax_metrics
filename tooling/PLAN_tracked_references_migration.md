# Plan — retire `golden`/`main_baseline`, move to tracked references

**Status:** decisions resolved 2026-06-18; implementing. Phases 1–2 of the Option-A migration are
done (per-geometry single-device fallback; `main` + `prerelease` added to `TRACKED_BRANCHES`).
This covers the remaining **Phase 3** (dashboard references from tracked runs), **Phase 4** (engine
+ repo cleanup / deletion), and a small independent **Phase 0** (harness UX: regression ≠ failure).
Scratch doc — relocate to the sharding plans dir or delete once the migration lands.

**Resolved decisions (2026-06-18):** A → **option 1** (prior-run hard gate; best-ever as the
visible drift check). B → **option 1** (vs-main is a dashboard overlay only; engine reads no
cross-branch files). C (correctness arrays, surfaced below) → **option 1** (keep the fingerprint
gate; delete both dormant array systems).

**Live validation (2026-06-18):** the first tracked `main` + `prerelease` CPU runs both hard-gate
`FAIL` — and *every* hard hit is `[golden] … is_sharded True -> False` (main: 30, prerelease: 12
cone-only). Cause: golden was captured on the dev sharding branch (`greg/conebeam_sharding`,
v0.6.18), so a non-/partially-sharding branch trips a structural change on every cell, and these
branches have no prior same-branch run, so golden was their *only* reference. This is exactly the
false-alarm class the migration removes (post-Phase-4: first run = cold-start `warn`, never `fail`;
later runs gate against their own prior). Strengthens the case for doing Phase 4 promptly.

---

## Goal

Remove the hand-blessed snapshot references (`golden/golden_<plat>.yaml`,
`golden/main_baseline_<plat>.yaml`, the `*.npy` arrays, and the two `capture_*.py` scripts) and
replace them with references derived from the **tracked runs** the harness already produces:
the latest `main` run, the latest `prerelease` run, the branch's own **prior** run, and
**best-ever** per cell. No more "remember to recapture golden" step.

---

## What golden does today (so we know what we're replacing)

Three reference tiers feed the system (engine map confirmed):

| Tier | Source | Role | Gated? |
|---|---|---|---|
| **prior run** (same branch) | `_find_prior()` over `out_dir` run files | commit-over-commit drift | **HARD** (ok→fail, GPU mem >8%, structural, correctness) + soft (time/speedup) |
| **golden** | `golden/golden_<plat>.yaml` | hand-blessed "expected state" anchor | **HARD** (same rules as prior) |
| **main_baseline** | `golden/main_baseline_<plat>.yaml` (n=1 cells) | "did sharding add 1-device overhead vs released main" | **SOFT** only (`vs_main` notes) |

Key facts that make this tractable:
- **`gate_run(result, references, config)`** (`performance_tracking.py:735`) is reference-agnostic
  — it compares against whatever `(label, ref)` tuples it's handed. **Cold start (no refs) is
  all-soft, never a fail** (`:744`). So dropping golden from the ref list is safe: the gate keeps
  working on prior-run alone.
- The **`.npy` files are written but never read** by either the engine or the dashboard (confirmed).
  Pure dead weight — safe to delete with zero functional impact.
- **best-ever** (`M.records[plat|branch]`) is *already* derived from the tracked runs and overlaid
  on the dashboard. It's the proof-of-concept that run-derived references work.

---

## Decisions (resolved 2026-06-18)

### Decision A — what anchors **slow drift** once golden's hard gate is gone? ⚠️ load-bearing — **DECIDED: option 1**

Golden gave one thing prior-run gating can't: a **fixed anchor**. Prior-run gating is
commit-over-commit, so a slow leak (each commit +2%, never tripping the 25%/8% thresholds) can
ride the baseline up forever and never fire. Golden caught that by comparing to a frozen point.

Options once golden is removed from the gate:
1. **(Recommended) Prior-run hard gate only; best-ever as the *visible* drift check.** The gate
   stays commit-over-commit (HARD). Slow drift shows up as the current curve pulling away from the
   best-ever overlay on the dashboard — visible, not alarmed. Simplest; no new gate logic. Revisit
   if real drift slips through unseen.
2. **Add a SOFT best-ever gate.** Same as (1) but the engine also emits a soft note when a cell is
   >X% slower than its best-ever. Catches drift in the YAML/email too, still never a hard fail
   (best-ever is noisy — one lucky-fast run sets an unbeatable bar). ~20 lines in the engine.
3. **Keep a pinned baseline run** (e.g. a tagged `prerelease` run) as a hard anchor. This is golden
   under a new name — re-introduces a "bless this run" step. Not recommended; defeats the purpose.

My recommendation: **(1)** now, with **(2)** as an easy follow-up if we want drift in the alert path.

### Decision B — does the run YAML still carry a machine-readable "vs main" note? — **DECIDED: option 1**

Today the engine writes `result["vs_main"]` (soft notes vs the `main_baseline` snapshot). After
migration the equivalent comparison ("dev vs latest tracked main/prerelease") can live in two places:
1. **(Recommended) Dashboard only.** The dashboard already holds all runs in memory; it derives
   "latest main run" by filtering `runs`. The cross-branch comparison becomes an overlay there.
   The engine stops reading any cross-branch file → simplest engine, no new coupling.
2. **Engine emits it too.** The engine reads `results/<plat>/main/*.yaml` (cross-branch, within the
   stable results tree) to write a `vs_main`/`vs_prerelease` soft note into each run YAML, for the
   nightly log/email. More self-contained YAML, but adds "engine reads other branches' results."

My recommendation: **(1)** — the hard gate (prior-run) is the alert; cross-branch is human-context,
which belongs on the dashboard.

> Note on cross-branch gating: comparing a dev cell to a `main`/`prerelease` cell is comparing
> *different code*, so a delta there isn't necessarily a regression. That's why main/prerelease are
> proposed as **overlays/soft context**, not hard-gate references. The hard gate stays same-branch
> (prior-run). Flagging in case your mental model had the gate firing on vs-main.

### Decision C — the dormant array correctness machinery — **DECIDED: option 1**

Investigation (2026-06-18) found the array-level "deep-diff" correctness gate was **never wired up**:
- `capture_main_baseline.py` writes `golden/<geom>_<op>.npy`, but nothing ever loads them — the
  comparator (`compare_to_baseline`, referenced in comments as "deferred") **does not exist**.
- A second, older system in `scaling_common.py` (`save_baseline`/`load_baseline`,
  `baselines/<op>.npy`) has **zero callers** and no data on disk.
- The **active** correctness gate is the per-cell **fingerprint** (shape/dtype/robust
  aggregates/samples/padding-zero) embedded in each run YAML — HARD on shape/dtype or aggregate
  beyond `rtol`. That stays.

**Option 1 (chosen):** keep the fingerprint gate; delete both dormant array systems
(`golden/*.npy`, and the orphaned `save_baseline`/`load_baseline`/`BASELINES_DIR` in
`scaling_common.py`). Accepts a small blind spot (an error that changes the array element-wise while
preserving its aggregates). Folded into Phase 4 deletions.

---

## Phase 0 — harness UX: a regression is not a harness failure (independent; do first)

Surfaced by the 2026-06-18 run. The loop already does **not** abort on a gate fail
(`run_regression.sh` has no `set -e`; the engine call is `if … else GATE_FAIL=1`, and `exit 1` is
deferred to the end — line ~239). The only wart: a manual run reports a perf regression (a
*successful run that found something*, exit 1) identically to a broken harness (failed
clone/install/push → exit 2), and the wrapper labels it "failed" + pauses — twice (both
`run_regression.sh`'s trap and `run_one_night.sh`'s fire).

Exit codes are *already* separated (`2` = FATAL setup/transport; `1` = gate regression; `0` =
clean). Fix is just the messaging:
- **`action_scripts/run_one_night.sh`** — trap branches on `rc`: `1` → "completed — regression(s)
  DETECTED (exit 1); this is an alert, not a failure; results were recorded + pushed" (+single
  pause); `≥2` → "FAILED (exit N) — harness/setup error" (+pause). Invoke `run_regression.sh` with
  `</dev/null` so its own interactive trap doesn't install → one clean message + one pause.
- **`tooling/regression/run_regression.sh`** — its interactive trap (line ~19–21) branches the same
  way, so a *direct* terminal invocation is also correctly worded.
- Unattended path unchanged: no tty → no trap/pause; `exit 1` still propagates → cron/slurm mail.

---

## Phase 3 — dashboard references from tracked runs (do now; code-only, safe, reversible)

> **DONE + verified 2026-06-18 (headless Chrome, real cpu data).** Simplified vs the original
> sketch: the JS already had `latestRun`/`currentRun`/`findCell`, so the overlays are derived
> straight from `M.runs` — **no `build_dashboard.py` change was needed**. Changes were JS + template
> only: `dashboard.js` (new `refRun()`; `refVal`/`refSeries`/`refProvenance`/`REF_LABEL` read tracked
> runs; dropped the `main`→n=1 special-case; reworded the gate-detail fallback) and `template.html`
> (selector → `none / main / prerelease / prior / best-ever`). Verified: all four refs resolve, with
> correct provenance + value lookups, zero boot errors. `build_dashboard.py` still *emits* the now-
> unused `golden`/`main` JSON keys → that dead emission + `_parse_baseline` move to **Phase 4**.

Goal: the "compare against" overlays come from tracked runs, not `golden/` files. Safe to do before
any `main`/`prerelease` data exists — those options simply show "no data" until today's runs land,
then populate on the next Pages rebuild. Golden files stay on disk, harmlessly unused, until Phase 4.

**`tooling/viewer/build_dashboard.py`**
- In `collect_data()` (~`:206`): stop calling `_parse_baseline(golden_dir/…)` (~`:237–244`).
  Instead derive, per platform, from the `runs` list already collected:
  - `mainRun[plat]`   = latest run with `git_branch == "main"` (by commit-time tag)
  - `prereleaseRun[plat]` = latest run with `git_branch == "prerelease"`
  - (prior-run and best-ever already exist: prior is per-branch via run sort; `records` = best-ever.)
- Reuse the existing cell-flattening (`_cell` / the dict keyed `geom|op|size|ndev` with
  `min_ms`/`mem_mb`/`fp`) so the overlay lookups in JS are unchanged in shape.
- Emit new top-level keys (replacing `"golden"`/`"main"`): `"mainRun"`, `"prereleaseRun"`
  (each `{plat: {branch, commit, commit_date, version, cells}}`, same shape `_parse_baseline`
  produced, so JS changes are minimal).

**`tooling/viewer/dashboard.js`**
- `REF_LABEL` (~`:32`): `{ main: "main", prerelease: "prerelease", prior: "prior run", best: "best-ever" }`.
- `refVal()` (~`:221`): `main` → `M.mainRun[plat]`; `prerelease` → `M.prereleaseRun[plat]`;
  `prior` → the current branch's prior run (derive once per render from `runs`); `best` → unchanged.
- `refSeries()` (~`:229`): **drop the `state.ref==="main" ? [1] : ndevs` special-case** (~`:231`) —
  tracked runs carry the full device grid, not n=1-only. (Single-device-only geometries already
  contribute just n=1 cells, so `spanGaps`/null handling already covers them.)
- `refProvenance()` (~`:52`): read provenance from `M.mainRun`/`M.prereleaseRun`/prior instead of
  `M.golden`/`M.main`.
- Default `state.ref` (~`:29`): keep `"none"`.

**`tooling/viewer/template.html`**
- Selector (~`:43–48`): `none / main / prerelease / prior / best-ever` (drop the "baseline"=golden
  option). Values: `none|main|prerelease|prior|best`.

**Verification (Phase 3):** rebuild via `action_scripts/build_dashboard.sh`; headless-Chrome dump.
Before today's runs land: main/prerelease overlays empty (expected). After the cpu+gpu
`run_one_night` runs land and a rebuild runs: selecting `main`/`prerelease` overlays the tracked run;
provenance string shows branch@commit·date. **This is the gate for starting Phase 4** — we want to
*see* tracked references working on the live dashboard before deleting golden.

---

## Phase 4 — engine + repo cleanup / deletion (after Phase 3 verified with real data)

**`tooling/scaling_tests/performance_tracking.py`**
- Remove golden from the ref list (~`:958–972`): drop the `golden_base`/`golden_path` resolution and
  the `refs.append(("golden", …))`. Keep the `prior` append. `gate_run` is unchanged (it just gets
  one fewer ref).
- Remove the `main_baseline` soft-note block (~`:974–982`) and delete `main_perf_notes()` (~`:787`)
  + the `vs_main` emission. *(Decision B = option 2 would instead repoint this at the tracked main
  run; option 1 deletes it.)*
- Remove Config fields `golden_path`, `golden_dir`, `main_baseline_path` (~`:97–102`).
- Remove `_parse_baseline()` (engine copy, ~`:175–200`), `capture_golden()` (~`:1021`),
  `_merge_golden()` (~`:1006`).
- *(Decision A = option 2: add the soft best-ever gate here — ~20 lines comparing each cell to
  `records`-equivalent best, emitting soft only.)*

**`tooling/scaling_tests/scaling_common.py`** — remove `golden_dir()` (~`:249–265`).

**`tooling/scaling_tests/run_nightly.py`** — remove the `REG_GOLDEN_DIR` read + `golden_dir`
override (~`:48–50`).

**`tooling/regression/run_regression.sh`** — remove the `REG_GOLDEN_DIR=…` env (~`:190`) and update
the "golden + vs-main baseline come from the metrics repo's golden/" comment (~`:185–186`).

**`tooling/viewer/build_dashboard.py`** — remove the now-dead `_parse_baseline()` and the `golden_dir`
constant (the Phase-3 rework already stopped calling them; Phase 4 deletes the dead code).

**Deletions:**
- `tooling/scaling_tests/capture_golden.py`
- `tooling/scaling_tests/capture_main_baseline.py`
- `golden/` entirely (4 `*.yaml` + 8 `*.npy`).
- `tooling/scaling_tests/backfill_commit_dates.py` — its migration is complete and it references
  `golden/*.yaml`; delete it too (confirm first it's not referenced elsewhere).

**Docs to reword:**
- `README.md` (root) — layout block describing `golden/` (~`:38–40`).
- `tooling/regression/README.md` — `:5` (diffs against golden), `:15`/`:19` (capture scripts +
  golden/ layout), `:77–79` (the "longer-term plan is to replace these snapshots…" note — that plan
  is now *done*).
- `tooling/viewer/README.md` — `:30–31` (golden/main rows), `:40` (the selector description).
- `action_scripts/add_run.sh` comments (~`:14/46/91`) mention "no golden gate" — reword to "no gate"
  (the `REG_GATE=0` behavior is unchanged; just the wording).

**Verification (Phase 4):**
- `action_scripts/run_one_night.sh` (or `REG_SMOKE=1`) green; a run YAML still gets a `gate` block
  with `compared_to: ["prior:…"]` (golden absent), `result: pass/warn/fail` as appropriate.
- Cold start (no prior) still returns `warn`, never `fail` (`gate_run` `:744`).
- `grep -ri golden tooling/ action_scripts/ README.md` → only intentional history, if any.
- Dashboard rebuild clean; the Pages Action still green (no golden/ dependency).

---

## Risks / rollback

- **Slow-drift blind spot** — see Decision A. The one real capability change. Mitigated by the
  best-ever overlay (option 1) or a soft best-ever gate (option 2).
- **Phase ordering** — Phase 3 is reversible (dashboard-only; golden files untouched). Phase 4 is the
  irreversible deletion; gate it on *seeing* main/prerelease overlays populate on the live dashboard.
- **`prerelease`/`main` may lag** — if a tracked branch hasn't been measured recently, its overlay is
  stale (shows the last commit-time it ran). The provenance string (branch@commit·date) makes that
  explicit, which is the honest behavior.

---

## Checklist

- [x] Decision A (slow-drift anchor) — **1** (prior-run hard gate; best-ever visible)
- [x] Decision B (vs-main note location) — **1** (dashboard overlay only)
- [x] Decision C (correctness arrays) — **1** (keep fingerprint; delete dormant array systems)
- [x] Phase 0: run_one_night.sh + run_regression.sh trap — regression (exit 1) ≠ failure (exit ≥2) — done + verified 2026-06-18
- [x] Phase 3: dashboard.js refRun/refVal/refSeries/refProvenance/REF_LABEL — done
- [x] Phase 3: template.html selector (none/main/prerelease/prior/best) — done
- [x] Phase 3: rebuild + verify overlays (cpu main/prerelease/prior, headless Chrome) — done 2026-06-18
- [ ] Phase 3: (n/a) build_dashboard.py overlays — unnecessary; runs already carry the cells
- [ ] Phase 3: re-verify on gpu once gpu main/prerelease runs land (same code path)
- [ ] Phase 4: engine golden/main_baseline removal (+ optional soft best-ever gate)
- [ ] Phase 4: scaling_common/run_nightly/run_regression.sh REG_GOLDEN_DIR removal
- [ ] Phase 4: delete capture_*.py, golden/, backfill_commit_dates.py
- [ ] Phase 4: docs reworded
- [ ] Phase 4: run_one_night green + grep clean
