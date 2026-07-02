# Dependency canary — treat a jax/jaxlib change like a commit to `main`

**Status:** design (not yet implemented).  **Motivation:** jax/jaxlib 0.10.2 slowed GPU forward
projection ~2–3× with byte-identical mbirjax code; the nightly missed it for days because its dedicated
env is *sticky* (`pip install -e` never upgrades already-satisfied deps) and fire-on-change triggers only
on a branch-tip move.  We want a dependency regression caught promptly, with clean attribution
(dependency vs code).

## 1. Core idea

Model each `(platform, branch)` timeline as a sequence of runs, each a pair **`(commit, dep-set)`**, with
the invariant that **adjacent runs differ in exactly one axis** — code *or* dependencies, never both.  The
gate already compares each run to its immediate predecessor (`_find_prior`), so that invariant keeps every
gate step a clean single-variable signal.

Treat a **jax/jaxlib version change** as a first-class event, like a commit.  Use `main` as the canary
(stable, slow-moving; once the prerelease→main PR lands it also exercises the sharded ops, so it's fully
representative).  When the resolved jax/jaxlib for `main` changes:

- **dep change, no new commit** → re-run `main`'s current tip `C1` with the new deps → gate isolates the
  **dependency** delta (`C1/D1 → C1/D2`).
- **dep change AND a new commit `C1→C2`** → two runs, in this order (Greg's decomposition):
  1. `C1` (the *previous* tip) with the new deps → `C1/D1 → C1/D2` = pure **dependency** delta,
  2. `C2` (the new tip) with the new deps → `C1/D2 → C2/D2` = pure **code** delta.
  Re-running the *previous* commit is the key: it's the code that already has an old-deps baseline to diff
  against.

`main` here is the initial canary; the same mechanism can later be enabled for `prerelease` (a knob).

## 2. Trigger — the jax/jaxlib fingerprint

Scope the trigger to **`jax` and `jaxlib` only** (the compute-central deps; numpy/scipy/etc. release often
and don't move time/memory materially — including them would fire the canary constantly for no signal).

- **Signal:** the newest `jax`/`jaxlib` on **PyPI** (unfiltered), obtained cheaply from PyPI JSON
  (reuse/extend `check_jax_release.py`).  No clone or resolve needed.
- **State:** `state/<plat>/jax_seen` = the PyPI-latest we last fired on.  **Fire when
  `pypi_latest != jax_seen`**; after firing, set `jax_seen = pypi_latest`.
- **The install applies the exclusion,** not the check.  So a newly-released *excluded* version (e.g.
  `0.10.2`) fires **one** run that resolves back to the last good version — a harmless no-op (gate quiet,
  toolchain shows the old version) — and `jax_seen` prevents it re-firing.  This is rare in practice: we
  only exclude a version *after* measuring it bad, by which point it isn't "new."  (Storing the *installed*
  version instead of `jax_seen` would re-fire nightly while an excluded version is PyPI-latest — hence
  `jax_seen`.)  The actual installed versions live in each run's `toolchain` block for provenance.
- CPU and GPU keep disjoint state (per-platform), so their canaries are independent.

## 3. Multiple runs per commit — the one real plumbing change

Today a commit maps to exactly one run file, `regression_<plat>_<commitUTC>_<sha8>.yaml`, and the timeline
sorts by that name (= by commit time).  The canary produces **two runs of the same commit** (`C1/D1`,
`C1/D2`) which would collide on that name.  Fix: append a **dependency-generation counter**.

- **State:** `state/<plat>/depgen` = a monotonic integer = **the identity of the current dep set**.  Bump
  it whenever the about-to-install set differs from the last installed — so a *single night* can advance it
  more than once: e.g. `(J1,O1) → (J2,O1)` at the targeted jax upgrade, then `→ (J2,O2)` at the full
  upgrade.  Steps that install the same set share one gen (e.g. the jax step and code step below are both
  at the `(J2,O1)` gen); each run is stamped with its set's gen.
- **Filename:** `regression_<plat>_<commitUTC>_<sha8>_g<NNNN>.yaml` (zero-padded, e.g. `_g0007`).
  - Ordering: lexicographic filename sort is still **commit-time primary**, with `_g<NNNN>` a tiebreaker
    *within* a commit, so `_find_prior` returns `C1/D1` as the prior of `C1/D2`, and `C1/D2` as the prior
    of `C2/D2`, unchanged in logic.
  - Idempotency: same commit + same gen → same filename → a re-measure overwrites (no duplicate points); a
    dep change bumps the gen → a new file.
- **Run doc:** add `dep_gen: <int>` and `run_reason: "commit" | "dep-change"` for provenance and so the
  dashboard can tell the two apart.
- The *first* run of a commit under the current single-run world can keep `_g0000` implicitly (parsers
  should treat a missing suffix as gen 0), so existing files need no rename.

## 4. Install — upgrade jax to latest; the editable install enforces the exclusion

`pip install -e "$WT[extras]"` won't bump jax (an installed 0.10.1 already satisfies `!=0.10.2`).  On a
jax-change night, before the branch loop, force it to the latest — jax/jaxlib only, so numpy/scipy aren't
perturbed:

```
pip install -U "jax[<extra>]" jaxlib          # <extra> from EXTRAS: cuda12 / cuda13 / (none on cpu)
```

**No explicit `!=0.10.2` is needed:** each branch's subsequent editable install re-resolves against that
branch's pyproject and **downgrades an excluded version back to the last good one**, so the exclusion
stays single-sourced in pyproject.  (This is also *why* an excluded release is a quiet no-op — upgrade to
it, then the install pulls it back down.)

**Shared-env note (important):** the dedicated `mbirjax_regression` env is shared across all tracked
branches (sequential installs), so this jax upgrade updates the env for **every** branch that night, not
only the canary.  That is what we want — "don't let it get stale," the whole system adopts the latest
jax.  The canary (`main`, same code across the jax boundary) is what supplies the clean jax-vs-code
*attribution*; the other branches simply move to the new jax and, on a night their own code also changed,
show a combined delta that the canary + `toolchain` field disambiguate.  (This is the status quo — a
shared env that occasionally jumped jax via a scratch wipe — now made deliberate and attributed.  A truly
isolated per-branch jax would need a separate env per branch; deliberately out of scope.)

## 5. Three axes and the full-dependency refresh

There are three change axes; each gets its own single-variable step, run in a fixed order so the gen/prior
chain stays clean (jax → code → everything-else):

- **NJ** — new jax/jaxlib (precise fingerprint, §2).  Step **jax**: previous tip `C1` with `(J2,O1)`
  (targeted jax upgrade, others held).
- **NC** — new `main` commit.  Step **code**: new tip `C2` with the same `(J2,O1)`.
- **FULL** — periodic refresh of *all other* deps.  Step **other-deps**: current tip with `(J2,O2)`
  (full eager upgrade).

**The other-deps trigger is a MAX-STALENESS TIMER, not change-detection.**  Detecting "any other dep
changed" means an all-deps resolve/fingerprint and would fire on every numpy/scipy/matplotlib release for
usually-no signal.  Instead keep `state/<plat>/last_full_refresh` and set `FULL` when
`now − last_full_refresh > DEP_FULL_REFRESH_DAYS` (default 14).  This bounds staleness to ≤ N days
regardless of main/jax activity, and a timer-fired step that finds nothing new is a cheap no-op with a
quiet gate.  When a jax/code night is already happening, the full step **piggybacks** (no extra spin-up);
in a quiet stretch it fires **standalone** on the current tip → `C1+(J1,O2)` vs `C1+(J1,O1)` = a clean
other-deps delta.  (`NO`-detection is deliberately dropped; only jax keeps a precise fingerprint.)

### Fire logic (`run_regression.sh`), per platform

```
NJ   = latest_allowed(jax,jaxlib) != state/<plat>/main.deps          # precise fingerprint (§2)
FULL = (now - state/<plat>/last_full_refresh) > DEP_FULL_REFRESH_DAYS

for BR in TRACKED_BRANCHES:
    tip = ls-remote(BR);  last = state/<plat>/<BR>;  NC = (tip != last)

    if BR == DEP_CANARY_BRANCH (main):
        deps = sticky
        if NJ:   bump depgen; deps=(J2,O1);  measure(commit=last, deps, gen)  # 1 jax   : C1+(J2,O1)
        if NC:                               measure(commit=tip,  deps, gen)  # 2 code  : C2+(J2,O1)
        if FULL and (NJ or NC or timer-due): bump depgen; deps=(J2,O2)
                                             measure(commit=tip,  deps, gen)  # 3 others: tip+(J2,O2)
                                             write state/<plat>/last_full_refresh = now
        write state/<plat>/main       = tip
        write state/<plat>/main.deps  = actual installed jax/jaxlib
    else:
        if NC: measure(commit=tip, deps=sticky)                               # unchanged branches
        write state/<plat>/<BR> = tip
```

Notes: order is fixed 1→2→3 so each step's prior is the immediately-preceding step (`_find_prior` +
`depgen`).  `measure(commit=last, …)` re-checks out the *previous* tip so the jax step diffs against the
existing `C1+(J1,O1)` baseline.  When `NC` is false, "tip" == `last` == `C1`, so steps still chain
correctly (jax on `C1`, then other-deps on `C1`).  A failed step (OOM/error) records the failure cell and
the sequence continues.  `bump depgen` happens at each dep-set *transition*, so steps 1&2 share a gen and
step 3 gets the next (§3).

## 6. Attribution

Nothing new needed for the numbers — the `toolchain` block already records `jax`/`jaxlib` per run, so a
dep step's toolchain differs from its predecessor's and the gate delta is directly attributable.  The
`dep_gen` + `run_reason` fields make it explicit ("this point is dep-generation 7, triggered by a dep
change").

## 7. Dashboard (`build_dashboard.py`, `dashboard.js`)

- **Run key:** include `dep_gen` so two runs of the same commit are distinct (today the key is
  commit+dates → they'd dedupe/collide).
- **History x-position:** plot a dep-change run at its **measurement** time, not the commit time (both C1
  runs share a commit time and would stack).  Add a fine `measured_at` timestamp to the run doc if the
  day-granular `date` isn't enough.
- **Labeling:** show the jax/jaxlib version (from `toolchain`) on the point / tooltip, and mark
  `run_reason == "dep-change"` distinctly (e.g. a small badge) so a dep step reads differently from a code
  commit.
- **records / vs-main:** keyed on commit today; with two runs per commit, define the tiebreak (latest
  `dep_gen` wins for "best-ever" and for the vs-main reference).

## 8. Affected files

| file | change |
|---|---|
| `run_configs.env` | `DEP_CANARY_BRANCH="main"`, `DEP_CANARY_ENABLED=1`, `DEP_FULL_REFRESH_DAYS=14`, (later) opt-in list |
| `state/<plat>/` | new markers: `main.deps` (jax/jaxlib fingerprint), `depgen` (counter), `last_full_refresh` (epoch) |
| `tooling/regression/run_regression.sh` | dep fingerprint + gen bump; canary fire logic (§5) incl. the timer; call the jax/jaxlib (targeted) and full (eager) upgrades |
| `tooling/regression/lib_env.sh` | a `reg_upgrade_jax` helper (§4) |
| `tooling/regression/check_jax_release.py` | reuse/extend for the resolve-latest-allowed fingerprint |
| `tooling/scaling_tests/performance_tracking.py` | `_g<gen>` in the output filename; `dep_gen`/`run_reason`/`measured_at` in the run doc; `_find_prior` already tolerant (verify) |
| `tooling/dashboard/build_dashboard.py` | run key incl. `dep_gen`; measurement-time x for dep runs; records/vs-main tiebreak |
| `tooling/dashboard/dashboard.js` | plot/label dep runs; tooltip shows jax/jaxlib |
| `tooling/regression/README.md` | document the canary + the new state files |

## 9. Edge cases

- **No same-commit prior** (first canary run for a commit): the dep step's gate falls back to whatever
  prior exists (maybe a different commit) — not a clean dep delta, but self-corrects next time.  Optional:
  suppress the gate verdict (record-only) when there's no matching same-commit baseline.
- **Both CPU + GPU:** independent per-platform state/gen; a dep change fires both canaries with their own
  gen counters.
- **Recovery:** if a bad jax is later added to the pyproject exclusion, the resolve returns an older
  version → a dep change → a canary run that shows the recovery (fast again).
- **Frequency:** with the jax/jaxlib-only scope, this fires only on a jax/jaxlib release (~weeks apart),
  so the extra cost is small — one (or two) `main` runs per jax release.

## 10. Alternative considered — a separate `deps-canary` series

Instead of extra runs *on main's timeline*, put them in their own branch dir
(`results/<plat>/deps_canary/`).  Different commit times ⇒ different filenames ⇒ **no per-commit
collision, no filename/dedup ripple** (§3 and most of §7 disappear).  Cost: it's a separate series rather
than points on `main`, and the "same code / deps change" pairing lives within that series.  Chosen model
is on-`main` per the request; note this fallback if §3's plumbing proves heavier than expected.

## 11. Phasing

0. **Engine plumbing** (this PR): Config `dep_gen`/`run_reason`, run-doc stamping + `_g<gen>` filename
   (gen 0 = no suffix), `_find_prior` tolerant of the suffix, and `run_nightly.py` reading them from the
   env so the shell can drive it.  Testable in isolation, no behavior change at gen 0.
1. Fingerprint + `state/<plat>/{main.deps,depgen}` + the **dep-only** canary path (no new commit).
2. The **both-change** decomposition (§5 jax + code steps).
3. The **timer-driven full refresh** (§5 step 3) + `DEP_FULL_REFRESH_DAYS` + `last_full_refresh`.
4. Dashboard: run key incl. `dep_gen`, measurement-time x, labeling, records/vs-main tiebreak.
5. Extend to `prerelease` (knob) if sharded-only dep coverage is ever wanted.
