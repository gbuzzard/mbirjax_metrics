# Partition-sequence study

Measures how the **granularity schedule** (`partition_sequence`) affects VCD convergence,
using real CT scans subsampled so many experiments run cheaply.  The whole pipeline lives
here and is driven by one file, [`config.yaml`](config.yaml) — add a dataset or an experiment
by editing YAML, no Python changes.

Study findings prose: `mbirjax/experiments/partition_sequence/partition_sequence_plan.md`.

## Pipeline

| stage | script | runs on | in → out |
|-------|--------|---------|----------|
| preprocess | `build_cache.py` | **cluster** | raw scan (`/depot`) → `cache_dir/<tag>.h5` (+ `.json` sidecar) |
| reconstruct | `run_study.py` | **cluster (GPU)** | cache → `output_dir/<label>.json` per run |
| present | `build_page.py` | **local** | `data/<round>/*.json` → `partition_sequence.html` |

Sequences are **indices** into granularity `[1,2,4,8,16,32,64,128,256]`; the last index
repeats for the remaining iterations (so the tail granularity is a first-class knob).

### Where things run and how data moves

The two heavy stages run **on gautschi** — the raw scans live on `/depot` and the recons need
a GPU. You build and **view the HTML page locally** (it just needs a browser). Assume you have
this repo checked out **both places** (locally and on the cluster) and the `mbirjax` conda env
on the cluster. Three kinds of data, three ways they move:

- **Caches** (`cache_dir`, on `/depot`) — shared and persistent; **never synced**. Cluster jobs
  read them directly, and they are reused across teammates, so you usually build none.
- **`config.yaml` + result JSONs** (`data/<round>/`) — travel through **git**: commit on one
  side, `git pull` on the other. (For a quick look you can instead `rsync` JSONs straight from
  cluster scratch into your local `data/<round>/`; see step 4.)
- **Run outputs** (`output_dir`, on cluster **scratch**) — large and **purgeable**; you curate
  the few JSONs worth keeping into `data/` (step 4). Recons (`.npy`, opt-in) also land here.

## Run your own sweep

1. **[local] Edit `config.yaml`.** Add/choose your datasets under `datasets:` (each is one
   `(source, downsampling)` pair with its **own** `detector_factor` / `view_factor`), and add
   an experiment under `experiments:` — the datasets to run, the candidate sequences
   (`name: [indices]`), the phases, and any overrides of the `defaults:` block. Commit and
   push so your cluster checkout sees the same config (or edit it directly on the cluster —
   just keep the two copies in sync).

2. **[cluster] Build caches — usually a no-op.** From your cluster checkout:

   ```bash
   cd .../mbirjax_metrics/experiments/partition_sequence
   git pull                                     # get your step-1 config
   python build_cache.py                        # builds only datasets MISSING from the depot cache
   # or, for a big/aligned dataset, as a batch job:  sbatch build_cache.slurm
   ```

   Existing datasets are already on the shared depot cache (see `MANIFEST.md` there), so this
   only does work for a genuinely new `(source, downsampling)`.

3. **[cluster] Run the experiment** (GPU batch job):

   ```bash
   sbatch --export=ALL,PS_EXPERIMENT=<name> run_study.slurm
   squeue --me                                  # watch it;  tail -f slurm-mbirjax-pseq-*.log
   ```

   Per-run trajectory JSONs land in `output_dir` (scratch). A console `=== SUMMARY ===` and a
   `<tag>_floor.json` are written at the end.

4. **[cluster] Curate + publish the results you want to keep.** Copy the JSONs out of scratch
   into this repo's `data/<new-round>/`, then commit + push:

   ```bash
   mkdir -p data/<new-round>
   cp <output_dir>/<tag>_*.json data/<new-round>/   # includes <tag>_floor.json (shapes+floor)
   git add data/<new-round> && git commit -m "partition sweep: <name>" && git push
   ```

   *Quick-look alternative (skip git):* from your **local** machine,
   `rsync -av gautschi:<output_dir>/'<tag>_*.json' data/<new-round>/`.

5. **[local] Register in the page and build it.**

   ```bash
   git pull                                     # get the JSONs you just pushed
   # add one line to config.yaml `page.datasets`:  {tag: <tag>, rounds: [<new-round>], targets: [...]}
   python build_page.py                         # writes partition_sequence.html
   open partition_sequence.html                 # view in your browser
   ```

   `build_page.py` runs **locally** (no GPU, no depot — just the repo + `pyyaml`); sino/recon
   shapes and the noise floor auto-derive from the JSONs, so a new dataset needs only
   `tag` + `rounds` + `targets` in the `page:` block.

### Local dry run (no cluster, no data)

Before touching the cluster, exercise the whole chain **locally** in ~1 minute with the
`synthetic` cube phantom (no data files, no GPU). Override the two storage paths to local
scratch so nothing tries to reach `/depot`:

```bash
export PS_CACHE_DIR=/tmp/ps/cache PS_OUTPUT_DIR=/tmp/ps/results
python build_cache.py synthetic
python run_study.py                       # default_experiment = synthetic_smoke
```

## Shared cache on data_depot

Preprocessed sinograms are cached at **`/depot/bouman/data/mbirjax_metrics/partition_sequence/cache`**
(long-term, group-writable, visible from gautschi).  `build_cache.py` **reuses** an existing
`<tag>.h5` there and only preprocesses when it is missing, so the common case does zero
preprocessing.  Each cache has a `.json` sidecar with its provenance (source, downsampling,
build date); see `MANIFEST.md` in that directory for the menu of what is already built.  To
add a new dataset to the shared set, build it (it lands in the cache dir) and update the
manifest.  Point elsewhere with `PS_CACHE_DIR` if you want a private cache.

## Config knobs

- `defaults:` — study parameters (iteration caps, stop %, NRMSE mask, seeds…); an experiment
  entry overrides any of them.
- `page.datasets[]` — `{tag, rounds, targets}` per rendered dataset; optional
  `floor`/`sino`/`recon` are legacy fallbacks (new runs carry these in the JSON).
- Env: `PS_CONFIG` (alternate config file), `PS_EXPERIMENT` (which experiment),
  `PS_CACHE_DIR` / `PS_OUTPUT_DIR` (override storage paths), `PS_TAGS` (build_cache subset).

## The page

`partition_sequence.html` is self-contained (vendored uPlot inlined).  Per dataset: a summary
table (iterations/seconds to each NRMSE target + peak memory), then NRMSE vs iteration (curves
nearly collapse — convergence per iteration barely depends on the schedule) and NRMSE vs wall
time (curves fan out by tail-granularity cost); noise floor dashed; hover a name to highlight;
slider truncates the time plot.  Open with `#force-visible` appended to the URL when viewing
through headless/hidden-tab tooling.
