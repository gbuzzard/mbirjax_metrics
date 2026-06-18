# action_scripts

Top-level entry points for this metrics repo. Each is a thin wrapper around the engine/harness in
`tooling/`; run them from the `mbirjax` conda environment (they try to activate it for you), and
each keeps the terminal open on a nonzero exit instead of closing it.

| script | purpose |
|---|---|
| `build_dashboard.sh` | Rebuild the static dashboard (`dashboard/index.html`) from the YAML time series. Run it after new results land, then open the file in a browser. Wraps `tooling/viewer/build_dashboard.py`. |
| `add_run.sh` | Measure a **specific mbirjax commit** and add it to the tracked time series — e.g. to seed an older prerelease run onto the timeline. `--local` measures the branch checked out in your current mbirjax repo (must have no uncommitted changes); `<ref>` measures a branch/tag/sha from `MBIRJAX_REPO` (default `../mbirjax`); no args prints help. The commit is checked out into a throwaway worktree (your tree is untouched). |
| `run_one_night.sh` | Run **one nightly pass** by hand — the faithful single invocation of the harness (`tooling/regression/run_regression.sh`): for each tracked branch whose remote tip moved, clone it, run the tests + the perf engine, write results, and push. Use it to verify the pipeline before enabling the scheduled nightly (which runs the same harness on a timer). |

See `tooling/viewer/README.md` (dashboard) and `tooling/regression/README.md` (nightly) for details.
