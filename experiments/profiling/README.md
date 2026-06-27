# Fine-grained projector profiling

Investigation into **where time and memory actually go inside the mbirjax projection
kernels** — a level below the coarse min-time + peak-memory the regression engine records.
The goal is to see what is worth instrumenting or optimizing, using JAX's profiler / static
analysis on both targets and NVIDIA's tools on the GPU.

- **Repo / branch:** `mbirjax_metrics`, branch `jax_profiling`. Exploration only — the
  harness (`tooling/`) and the library are **not** edited.
- **What it measures:** the sibling **`mbirjax`** library, via the `mbirjax` conda env's
  editable install (which points at the `Research/mbirjax` worktree — so we profile whatever
  that checkout has). The scripts reuse the engine's own input builders
  (`performance_tracking.make_model` / `make_sinogram` / `make_indices` / `to_device`) by
  putting `tooling/scaling_tests` on `sys.path`, so we measure the library exactly the way the
  nightly does — just with a profiler wrapped around the **warm** call instead of a bare timer.
- **Env:** `source ~/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax`
  (JAX 0.10.2; this Mac is CPU-only).

## Scripts

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

## How to run

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax
cd <mbirjax_metrics>
python experiments/profiling/trace_back_projection.py        # exp 1 (trace)
python experiments/profiling/static_cone_back_kernels.py     # exp 2 (static + cliff)
```

Run parameters are clearly-labeled constants at the top of each script (no CLI args, so a run is
reproducible from the file). To trace the **multi-device** path, set `N_DEVICES = 4` in
`trace_back_projection.py` — it sets `MBIRJAX_NUM_CPU_DEVICES` to match via `setdefault`, so 4
virtual CPU devices appear and the trace shows the thread-pool fan-out + reduce-scatter.

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

## Next: GPU (Gautschi H100)

The high-value GPU experiment is the **platform inversion**: on CPU the band kernel is faster;
the lesson reports the pixel kernel is **2.25× faster on GPU**. Porting exp 1 + exp 2 to the H100
would confirm it AND let `ncu` show the gather kernel's roofline / HBM utilization (the
efficiency question the Mac can't reach), and `nsys` show whether the multi-GPU band
reduce-scatter overlaps compute. Needs: a cluster allocation, and possibly installing Nsight
Systems/Compute + `tensorboard-plugin-profile` (gated — confirm before installing/running heavy).
Mind the documented throttle/NUMA confounds when timing on the cluster.
