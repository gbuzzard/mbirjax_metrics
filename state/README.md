# `state/` — fire-on-change markers

This directory is the nightly harness's **fire-on-change bookkeeping**: it records the last commit each
tracked branch was **measured** at, so a run can skip a branch whose tip hasn't moved.

```
state/<platform>/<branch-slug>     # plain text: the full git sha last MEASURED for that branch+platform
```

- One file per **platform** (`cpu`, `gpu`) × **branch** (slug = the branch name with `/` → `_`, e.g.
  `greg_sharding_extensions`). Each file contains a single commit sha and nothing else.
- **Read + written only by** [`tooling/regression/run_regression.sh`](../tooling/regression/run_regression.sh):
  each run does `git ls-remote` for a branch's tip and **skips it if the tip equals the stored sha**;
  after a successful measurement it writes the measured sha back (the LAST step, so a crash mid-run
  re-measures next time). The markers are committed + pushed alongside `results/`.
- **`action_scripts/add_run.sh` does NOT touch this directory** — a hand-seeded backfill writes to
  `results/` but is intentionally *not* recorded here, so the nightly's change detection is unaffected.

Implications when you're working on the harness:
- To **force a re-measure** of a branch, delete its `state/<plat>/<branch>` file (or commit a new tip).
- These are **data, not config** — don't hand-edit them to "fix" the dashboard; the dashboard reads
  `results/`, never `state/`.
- This on-disk `state/` is unrelated to the `ui_state` object in `tooling/dashboard/dashboard.js` (the
  client UI selection). Different layer, different lifetime — see
  [`.claude/dashboard_orientation.md`](../.claude/dashboard_orientation.md) §2.
