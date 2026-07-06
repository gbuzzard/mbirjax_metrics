# Partition-sequence study

Measures how the **granularity schedule** (`partition_sequence`) affects VCD convergence,
using real CT scans subsampled so many experiments run cheaply.  The whole pipeline lives
here and is driven by one file, [`config.yaml`](config.yaml) — add a dataset or an experiment
by editing YAML, no Python changes.

Study findings prose: `mbirjax/experiments/partition_sequence/partition_sequence_plan.md`.

## Pipeline

| stage | script | in → out |
|-------|--------|----------|
| preprocess | `build_cache.py` | raw scan → `cache_dir/<tag>.h5` (+ `.json` sidecar) |
| reconstruct | `run_study.py` | cache → `output_dir/<label>.json` per run |
| present | `build_page.py` | `data/<round>/*.json` → `partition_sequence.html` |

Sequences are **indices** into granularity `[1,2,4,8,16,32,64,128,256]`; the last index
repeats for the remaining iterations (so the tail granularity is a first-class knob).

## Run your own sweep

1. **Pick / add datasets** in `config.yaml` under `datasets:` — each entry is one
   `(source, downsampling)` pair with its **own** `detector_factor` / `view_factor`.
2. **Define an experiment** under `experiments:` — the datasets to run, the candidate
   sequences (`name: [indices]`), the phases, and any parameter overrides (defaults come
   from the `defaults:` block).
3. **Build the caches** (once per new dataset; existing ones are reused from the shared
   depot cache — see below):

   ```bash
   python build_cache.py                 # build any missing dataset in the config
   sbatch build_cache.slurm              # or as a batch job
   ```
4. **Run the experiment** on the cluster:

   ```bash
   sbatch --export=ALL,PS_EXPERIMENT=<name> run_study.slurm
   ```
5. **Publish** — copy the run JSONs you want to keep from `output_dir` (scratch, purges)
   into `data/<round>/`, add the dataset to the `page:` block of `config.yaml` (just
   `tag` + `rounds` + `targets`; shapes and the noise floor auto-derive from the JSONs),
   then:

   ```bash
   python build_page.py                  # writes partition_sequence.html
   ```

### Local dry run (no cluster, no data)

The `synthetic` dataset is a tiny cube phantom needing no data files.  Override the two
storage paths to local scratch and run the whole chain in ~1 minute:

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
