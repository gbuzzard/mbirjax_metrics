# action_scripts

Top-level entry points + config for this metrics repo. The scripts are thin wrappers around the
engine/harness in `tooling/`; run them from the `mbirjax` conda environment (they try to activate it
for you), and each keeps the terminal open on a nonzero exit instead of closing it.

| script | purpose |
|---|---|
| `build_dashboard.sh` | Rebuild the static dashboard (`dashboard/index.html`) from the YAML time series and open it locally. (The live site is rebuilt automatically by a GitHub Action; see the repo README.) Wraps `tooling/viewer/build_dashboard.py`. |
| `add_run.sh` | Measure a **specific mbirjax commit** and add it to the tracked time series — e.g. to seed an older prerelease run onto the timeline. `--local` measures the branch checked out in your current mbirjax repo (no uncommitted changes); `<ref>` measures a branch/tag/sha from `MBIRJAX_REPO` (default `../mbirjax`); no args prints help. Checks out into a throwaway worktree (your tree is untouched). |
| `run_one_night.sh` | Run **one nightly pass** by hand — the faithful single invocation of the harness (`tooling/regression/run_regression.sh`): for each tracked branch whose remote tip moved, clone it, run the tests + the perf engine, write results, and push. Use it to verify the pipeline before enabling the scheduled nightly. |
| `enable_nightly.sh` / `disable_nightly.sh` | Start / stop the **scheduled** nightly. Platform-aware: macOS uses a launchd agent; Gautschi uses a managed SLURM `scrontab` block (resources from `run_configs.env`'s `SLURM_*` knobs). Wrap `tooling/regression/{enable,disable}_nightly.sh`. |
| `create_token.sh` | One-time setup of the fine-grained GitHub PAT used for the unattended push. See `create_token_instructions.md`. |

**`run_configs.env`** — the run-time knobs you edit: `TRACKED_BRANCHES`, `INSTALL_EXTRAS_cpu/gpu`,
`CONDA_PYTHON`. The harness sources it (via `tooling/regression/regression.env`); edits propagate to
the nightly automatically (it pulls the metrics repo before each run). Harness infrastructure
(URLs, paths, credentials, schedule, the `ENABLED` kill-switch) stays in `regression.env`.

See `tooling/viewer/README.md` (dashboard) and `tooling/regression/README.md` (nightly) for details.
