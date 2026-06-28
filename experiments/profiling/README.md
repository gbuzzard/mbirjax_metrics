# Fine-grained projector profiling

> ⚠ **SOME NUMBERS IN THIS README ARE STALE (jax 0.10.2).** Profiling must run the **production env + its pins**;
> the early runs used jax **0.10.2**, since **EXCLUDED** for a backend-AND-kernel-specific regression (4×-slowed
> the cone *band* kernel on CPU and the cone *forward* kernel on GPU). Everything was re-measured on **0.10.1**
> (2026-06-27): GPU back findings **CONFIRMED**, GPU forward 599→148 ms, CPU cone-back corrected. This README is
> the methodology/history — for the **corrected, authoritative inventory** see [`key_findings.md`](key_findings.md).

Investigation into **where time and memory actually go inside the mbirjax projection
kernels** — a level below the coarse min-time + peak-memory the regression engine records.
The goal is to see what is worth instrumenting or optimizing, using JAX's profiler / static
analysis on both targets and NVIDIA's tools on the GPU.

- **Repo / branch:** `mbirjax_metrics`, branch `jax_profiling` (the tooling). The **harness
  (`tooling/`) is not edited**; the measured **`mbirjax`** library now carries lightweight
  `jax.named_scope` annotations (no behavior change — they only tag HLO op metadata) added for the
  gen-2 region attribution below, committed on the `mbirjax` side.
- **What it measures:** the sibling **`mbirjax`** library, via the `mbirjax` conda env's
  editable install (which points at the `Research/mbirjax` worktree — so we profile whatever
  that checkout has). The scripts reuse the engine's own input builders
  (`performance_tracking.make_model` / `make_sinogram` / `make_indices` / `to_device`) by
  putting `tooling/scaling_tests` on `sys.path`, so we measure the library exactly the way the
  nightly does — just with a profiler wrapped around the **warm** call instead of a bare timer.
- **Env:** `source ~/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax`
  (JAX **0.10.1** — the production pin; **0.10.2 is excluded**, see the banner; this Mac is CPU-only).

## GPU sync to git when local changes match remote:
```bash
git stash -u    # stashes the tracked mods + the 2 untracked new scripts; NOT the gitignored outputs
git pull
git stash drop  # safe to drop — the stashed content is identical to what you just pulled
```

## Scripts

The gen-1 scripts below answer **one-off questions**; **gen-2** (next section) produces a diff-able
region time series — the profiling analog of the regression YAMLs.

### Gen-1: per-question one-offs

| file | what it does | which question it answers |
|---|---|---|
| `trace_back_projection.py` | warm `jax.profiler.trace` of `model.sparse_back_project` (cone), with a self-time / per-track trace summarizer | **where wall-time goes across the 4 layers** (host orchestration · cross-device comms · compiled XLA program · innermost kernel) |
| `static_cone_back_kernels.py` | `lower().compile()` of the two cone back kernels (pixel vs band) → `cost_analysis` + `memory_analysis` + HLO dump; warm-time ablation across the cone CPU cache cliff | **working-set floor / FLOPs / logical bytes**, and **why** a kernel is slow (HLO structure) |
| `compile_time_projectors.py` | splits each op's compile into **trace → lower → compile**, plus cold/warm exec + jaxpr-eqn / HLO-line complexity | **where compile time goes** (relevant to the projectors.py batching-nest refactor) |
| `gpu_inventory.py` | step-0 cluster probe: H100 count, jax/jaxlib versions, `nsys`/`ncu`/`tensorboard-plugin-profile` availability, topology + idle throttle pre-flight | **what the GPU env has** before planning the heavyweight steps |

The four layers a single wall-clock number fuses together (back projection example):
1. **host orchestration** — `TomographyModel._back_project_all_bands` (thread pool, band loop, `device_put`);
2. **cross-device comms** — `sum_band_to_owner` (reduce-scatter), `assemble_sharded`;
3. **compiled XLA program** — the jitted scan/map/vmap nest in `projectors.py`;
4. **innermost kernel** — the per-view back kernel's gather / scatter-add.

## Gen-2: region-attribution time series (the before/after-a-redesign tool)

The gen-1 scripts answer one-off questions; **gen-2 produces a structured, diff-able region
breakdown** so a core redesign can be compared before/after. It rests on **`jax.named_scope`**
annotations in the library (e.g. `cone/back/pixel/vertical_fan`): these tag the **HLO op metadata**
with a stable, code-localized region name that survives XLA's fusion renames. The raw trace event
names stay XLA-named and rename across jax versions, so the **HLO is the bridge**. Pipeline:

1. **library `named_scope`** → each fusion's HLO `op_name` carries its authored region.
2. **trace** the warm jitted driver → per-fusion **self-time** (Perfetto JSON).
3. **lower the SAME driver** → HLO → **fusion-base → region** map.
4. **join** (trace self-time ⋈ HLO region) → **self-time per region** → one YAML **per platform**.

One file **per platform** by design: back projection's driver is path-dependent — CPU runs the band
kernel, GPU n=1 the pixel kernel — so the back regions are `cone/back/band/*` on CPU vs
`cone/back/pixel/*` on GPU. The divergence is visible, not averaged away.

| file | what it does |
|---|---|
| `region_attribution.py` | the trace⋈HLO **join**: `hlo_fusion_regions` (fusion base→region), `region_breakdown` (self-ms + % per region), `find_collisions` (bases spanning >1 scope) |
| `trace_utils.py` | shared trace parsing: `fusion_self_time` (exclusive self-time per fusion), `is_host_runtime` (host-frame filter) |
| `profile_measure.py` | **measure**: per cell, trace + lower the matching driver, join to regions, write `results/profile_<plat>_<commitUTC>_<sha8>.yaml` |
| `profile_diff.py` | **diff**: before/after region deltas for one platform (env-guarded — refuses cross-jax / cross-platform comparisons) |
| `profile_to_table.py` | **browse**: render a run as nested `GEOMETRY / OP / size / n=` YAML, regions sorted by share |
| `cuda_profiler.py` | `profiler_range()` (cudaProfilerStart/Stop via ctypes) for `ncu --profile-from-start off`; no-op on CPU |
| `debug_region_join.py` | diagnostic for "why does a region read ~0%?" — prints (A) collisions, (B) per-region HLO bases (in trace?), (C) collision detail |
| `run_gpu_all.sh` | the GPU step 0–7 driver (inventory · static · compile · traces · `profile_measure` · ncu), with the jax-pin guard |

**Two attribution subtleties, both load-bearing:**
- **Region `ms` is `pct × wall`, not raw self-time.** On CPU the intra-op TraceMe spans *overlap*, so
  raw self-times overcount (their sum ≫ wall). Each region's **share** (pct) is the stable quantity;
  `pct × wall` gives an absolute that sums to ~wall on both platforms.
- **Base-name collisions are surfaced, not hidden.** The join keys on the fusion *base* name — the
  instance suffix differs by backend (CPU `.N`, GPU `_N`) and must be stripped, or GPU kernels never
  match the HLO (→ ~68% unmapped). But stripping collapses two instances of the same base in
  *different* scopes into one key — a **collision** (e.g. GPU cone back-pixel has an accumulate
  `loop_add_fusion` in *both* fans). The trace's `_N` indices don't correspond to the HLO's (CUDA-graph
  capture renames kernels), so name-matching can't split them. So `find_collisions` reports any such
  base, `profile_measure` writes a per-cell `collisions:` field + prints `⚠`, and `profile_to_table`
  shows "up to X ms uncertain" — a collided region's number is never silently trusted. (The
  collision-proof long-term fix = read scope from XPLANE per-event metadata; deferred while collisions
  stay rare + bounded.)

## How to run

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax
cd <mbirjax_metrics>
python experiments/profiling/trace_back_projection.py        # gen-1 exp 1 (trace)
python experiments/profiling/static_cone_back_kernels.py     # gen-1 exp 2 (static + cliff)

# gen-2 region time series (the before/after-a-redesign tool):
python experiments/profiling/profile_measure.py              # measure  → results/profile_<plat>_*.yaml
python experiments/profiling/profile_to_table.py             # browse a run (nested geom/op/size/n)
python experiments/profiling/profile_diff.py                 # diff two runs (before/after), env-guarded
# on the GPU, run_gpu_all.sh drives steps 0–7 (inventory · static · compile · traces · profile_measure · ncu)
```

Run parameters live in one place — **`profiling.env`** (KEY=VALUE, like the regression's
`run_configs.env`), parsed by `profiling_config.py`; the scripts `from profiling_config import ...`
instead of each carrying a CONFIG block. Edit `profiling.env` to change the knobs for all scripts at
once. Sizes are **per platform** (CPU can't run the big GPU sizes; profiling has no cross-platform
cell, so they're independent — simpler than the regression): `SIZE_CPU=200x208x160` (the dashboard
CPU-max), `SIZE_GPU=512x448x384` (the next dashboard GPU size, where the band/capacity/multi-GPU
effects show). Each script resolves its size by detected platform via `profiling_config.size_for()`.
A same-named environment variable **overrides** the file for a one-off, and the **un-suffixed** `SIZE`
forces a size on *both* platforms, e.g. `SIZE=1024x1008x992 python experiments/profiling/profile_measure.py`
or `N_DEVICES_LIST=1,2,4 python …` (which sizes the CPU mesh automatically). Sizes are kept
**asymmetric** on purpose (symmetric sizes like 256³ can mask axis/stride effects).

Outputs (gitignore candidates — they're large/derived):
- `traces/<tag>/.../perfetto_trace.json.gz` — open at <https://ui.perfetto.dev>; the script also
  prints a self-time summary so you needn't open the UI for triage.
- `hlo/<geom>_<kernel>_<size>.txt` — the compiled HLO, for reading fusion structure by eye.

## What we learned (CPU, 2026-06-26)

**Exp 1 — cone back, 256³:**
- The attach point works with **zero harness/library change**: one `jax.profiler.trace` around
  the warm op gives a Perfetto trace + parseable JSON.
- Cone back on CPU is **gather-dominated** (the sinogram→cylinder gather), with a real
  secondary cost in **cone coordinate math** (`atan2`/`divide`/`cosine` — voxel→detector mapping).
- **The CPU trace resolves the compiled-program layer better than expected**: XLA:CPU emits a
  per-fusion TraceMe, so layers 3/4 get named, individually-timed fusions (not just host events).
- n=1 → n=4 is **1.21×** (the shared-CPU-bus bandwidth ceiling); the reduce-scatter +
  thread-pool orchestration are <1% — the op is compute-bound at every device count. (Matches the
  Phase D lesson in `mbirjax/.claude/lessons.md`.)

**Exp 2 — pixel vs band kernel, the cone cache cliff:**
- Cliff **direction reproduced**: below ~200³ pixel ≈ band (0.99×); at 256³ pixel is **2.05×
  slower** on CPU. (The lesson's ~8× is a *512³, driver-less* number; this is 256³ at the driver
  level — smaller is consistent; size-vs-driver not isolated.)
- **Cross-check:** the bare band driver at 256³ (11.8 s) ≈ exp-1's full *sharded* back at n=1
  (11.7 s) → the sharded orchestration adds ≈0 at n=1, independently reproducing
  "driver-less band loop ties the full sharded path at 1.00×".
- **Key finding — the XLA static counters point the WRONG way.** Every static metric ranks the
  *slow* pixel kernel as cheaper: fewer FLOPs (8.7 vs 24.7 G), fewer bytes (5.0 vs 29.6 GB), less
  temp (1.6 vs 6.6 GB) — yet it's 2× slower. The HLO shows why: the pixel path carries the
  documented `lax.map`+transpose (`f32[256,2048,128]` rolled buffers + 8 transposes), so its cost
  is **cache / access-pattern**, which `cost_analysis`/`memory_analysis` don't model.

**Exp 3 — compile time (cone, CPU):**
- Compile is **~size-invariant at ~0.25–0.35 s per op per shape**, dominated by XLA's HLO→executable
  phase (~0.17–0.24 s); the batching-machinery **trace+lower is ~0.08–0.12 s** (lowering ~0.03 s).
  Cross-check: the cold−warm gap ≈ the measured compile total.
- **The compile:run ratio flips with size** — at 128³ compile is ~25–35% of a cold call (warm run
  ~0.7–0.8 s); at 256³ it's <2% (warm run 12–24 s). So compile matters in the **small-problem /
  many-distinct-shapes / first-call** regime (VCD per-subset, tests, interactive), negligible for one
  big recon.
- **The cost is in the batching nest, not the kernel** — pixel vs band compile within ~10%; the jaxpr
  is **~1000 eqns for every op/size**. The real refactor lever is recompile *frequency* (the partial
  "remainder" batch makes compilation depend on `num_pixels/num_views % batch_size`); the next
  measurement for that project is **counting distinct compiles in a real VCD/test run** (`_cache_size()`).
- Caveats: single-shot compile timings (noisy at tens-of-ms); **HLO line count is non-monotonic in
  size** (XLA fusion choices) — use the jaxpr eqn count as the stable complexity proxy.

**GPU (2× H100, 2026-06-26) — step 1 (the three scripts ported as-is, n=1):**
- **Platform inversion CONFIRMED.** Cone back, pixel/band: CPU 2.05× (band wins) → **GPU 0.38× (pixel
  2.6× faster)**. Matches the lesson and re-justifies the GPU n=1 short-circuit to the pixel kernel.
- **Static-counter reliability is itself platform-dependent.** On GPU the counters AGREE with wall time
  (pixel does less work — 8.7 vs 19.6 GFLOP, 5.0 vs 8.9 GB, 1.1 vs 1.9 GB temp — AND is faster); the cache
  cliff that made them mislead is CPU-only. So `cost_analysis` picks the right kernel on GPU, the wrong one
  on CPU.
- **GPU back trace (n=1, pixel kernel):** ~69 ms/iter, ~100% GPU-compute-bound (compute stream 68.8 ms; host
  overlaps). Dominant kernels: **`loop_add_fusion` ≈40 ms (accumulate)** + **`loop_dynamic_update_slice_fusion`
  ≈17 ms (scatter-write)** = ~83%. Same op families as CPU, different XLA naming/balance; uses CUDA graphs.
  These two are the `ncu` roofline targets. (H100 ~350× faster than the Mac on pixel-256³: 69 ms vs 24 s.)
- **GPU compile is autotuning-dominated, heavy + noisy:** 59 ms → 2406 ms (CPU was a uniform ~0.25 s); the
  band kernel's first compile cost 2.4 s of autotuning. trace+lower stays ~0.1–0.27 s, so the batching nest
  is a SMALLER share of GPU compile — the refactor lever is reducing distinct autotuned kernels, not trace
  cost. Single-shot GPU compile timing is unreliable (autotuning variance).
- **Multi-GPU back is NON-MONOTONIC, and the trace pins the cause to the band kernel's transpose.** Cone
  back 256³: n=1 wall **72.8 ms** (pixel kernel, 1 GPU busy 69.5 ms) vs n=2 wall **94.7 ms** (band kernel,
  EACH of 2 GPUs busy ~91 ms). So 2 GPUs are 1.3× SLOWER than 1 — reproducing the lesson's n≈2.25 back
  crossover at 256³. Why: n≥2 drops the pixel short-circuit and runs the **band kernel**, dominated by
  **`input_transpose_fusion`** (~58 ms aggregate) — costlier per-GPU than the pixel kernel's accumulate
  (`loop_add_fusion` 40 ms) + scatter (`loop_dynamic_update_slice_fusion` 17 ms). The **NVLink reduce-scatter
  is cheap** (wall ≈ max stream busy + ~3.5 ms; D2D folds into the compute stream over NV18) — comms are NOT
  the limiter, the band kernel is. This is the "B4.5 lever" (make the band kernel GPU-competitive without
  the CPU cliff), now pinned to a specific fusion on real hardware → next ncu target: `input_transpose_fusion`.
- **`ncu` roofline (n=1 pixel kernel) refines "bandwidth-bound" → memory-ACCESS-PATTERN-bound.** The dominant
  accumulate kernel `loop_add_fusion_3` (2.05 ms/launch) runs at **96% Memory throughput but only 8% DRAM/HBM**
  and 29% L2 — it saturates the ON-CHIP memory path (L1/LSU/address generation from the scatter/gather), NOT
  HBM bandwidth. So there is no HBM headroom to chase; the lever is fewer/coalesced memory transactions.
  The scatter-write `loop_dynamic_update_slice_fusion` (0.44 ms) is instead **compute-bound (82% SM)**, 40%
  DRAM — a different target. Both at 82–95% occupancy (not launch/occupancy-limited). `cost_analysis` ("5 GB
  accessed") could not have distinguished these. Follow-ups: (1) `--set full` MemoryWorkloadAnalysis to name
  the exact saturated pipe on the accumulate; (2) ncu the band kernel's `input_transpose_fusion` (the
  multi-GPU limiter) — the current `ncu_back_projection.py` runs n=1 → pixel, so a band-path variant is needed.
- **`ncu` band kernel (`ncu_band_kernel.py`) — the multi-GPU limiter is L1/TEX-cache-bound.** The band kernel's
  dominant transpose fusions (~2.8–3.0 ms each: `input_transpose_fusion`, `..._1`, `input_cosine_transpose_fusion`)
  run at **99–100% L1/TEX-cache throughput** with HBM at only **6–17%** and compute at 13–44% — so the transpose
  access pattern saturates the L1/texture cache, NOT bandwidth, compute, or occupancy (74–98%). The cone
  coordinate math is even fused into a transpose (the cosine variant), also L1-bound. **Lever:** the band
  kernel's slowness is the transpose, with large HBM/compute headroom; restructure it to write pixel-like
  (`dynamic_update_slice`, no transpose) WITHOUT reintroducing the CPU cliff. The ncu data turns the "B4.5
  lever" from a guess into a measured target: relieve the L1 pressure from the transpose.

**Scoping conclusion — each tool answers a different question:**

| question | tool | on Mac? |
|---|---|---|
| where wall-time goes across the 4 layers | `jax.profiler.trace` (self-time + tracks) | ✅ |
| working-set floor / FLOPs / logical bytes | `cost_analysis` / `memory_analysis` | ✅ |
| *why* a kernel is slow (fusion barriers, materialization) | HLO `as_text` | ✅ |
| is the kernel microarchitecturally efficient (roofline, cache, occupancy) | `ncu` (GPU) / `perf` (CPU) | ❌ → H100 |

So **static analysis is the right ruler for capacity/FLOPs, the wrong ruler for kernel
efficiency** — "is the gather kernel at the bandwidth roofline" is inherently an `ncu` question.

## Measurement-hygiene gotchas

- **Trace only WARM iterations** — the first call(s) compile and would dominate an unfiltered trace.
- **Self-time, not naive sum** — naive per-name sums put wrapper events (`StepTraceAnnotation`,
  `block_until_ready`, the worker thread's lifetime) on top because they *contain* everything; the
  summarizer computes exclusive self-time and a per-track split instead.
- **At n>1 the per-fusion absolute seconds inflate** (overlapping TraceMe spans across device
  threads) — trust the **ranking**, the **track view**, and **wall time**, not absolute fusion seconds.
- **Static counters miss cache effects** (the headline above) — use wall-time ablation + HLO
  structure for the "why slow" question.
- **Backgrounded Python block-buffers stdout** to a pipe — incremental output won't appear until
  the process exits (the 256³ pixel kernel is minutes/call). Run in foreground, or `python -u`, to
  watch progress.
- These are CPU XLA fusions, **not** GPU kernels — op *families* should carry over, the *balance* won't.

## Next (GPU phase done; lower-priority follow-ups)

The core kernel hunt is **concluded** — the banner + [`key_findings.md`](key_findings.md) hold the
authoritative inventory (platform inversion confirmed on H100, the multi-GPU band-transpose limiter
pinned by `ncu`). Open items, roughly in priority order:
- **More `named_scope` coverage** — annotate the **parallel** and **qGGMRF** paths (only cone is
  scoped today) so the gen-2 region time series spans all geometries + the recon update.
- **`ncu --set full` MemoryWorkloadAnalysis** on the n=1 accumulate (`loop_add_fusion_3`) to name the
  exact saturated on-chip pipe (L1 vs LSU vs atomics).
- **512³ scale-up** and an optional **`nsys`** system timeline.
- **projectors.py batching-nest compile probe** — count distinct compiles in a real VCD/test run
  (`_cache_size()`) to size the recompile-frequency lever.
- **XPLANE-metadata attribution** — only if a collision ever bites materially (a large `collisions:`
  ms); replaces the perfetto-name⋈HLO join with collision-proof per-event scope metadata.

Cluster hygiene when re-measuring: mind the documented throttle/NUMA confounds, keep the **0.10.1**
pin (`run_gpu_all.sh` guards it), and the analysis-only extras (`tensorboard-plugin-profile`) install
on top of the shared env via `dev_scripts/install_tensorboard.sh` (which verifies jax/jaxlib unchanged).
