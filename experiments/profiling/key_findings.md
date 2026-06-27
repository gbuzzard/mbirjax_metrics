# Key findings — projector & prior bottleneck inventory

A scannable inventory of **bottlenecks** and **possible improvements** for
`{forward, back, qGGMRF}` × `{parallel, cone}`, on CPU and GPU (single- and multi-device).
This is the *inventory*; see [`README.md`](README.md) for methodology, the narrative arc, the
tool-to-question map, and how to run each script.

**Legend** — confidence: `✓` measured here · `~` partial (timing only / single size) · `?` hypothesis
(from `lessons.md`, not re-measured) · `—` not yet investigated.
**Platforms:** `CPU` (Mac, virtual devices) · `GPU1` (single H100) · `GPUn` (multi-H100, sharded).
Numbers are cone, 256³, warm, n=1 unless noted, from this investigation (2026-06-26).
**All `mbirjax/…` paths are in the sibling library repo** (`Research/mbirjax/`), not this repo.

---

## Code map (what the informal labels mean)

The user-facing call chain for back projection, and the precise functions each label refers to.
Forward and the prior follow the analogous chain.

| label used below | precise function (`mbirjax/…`) | what it is |
|---|---|---|
| **cone pixel-kernel** | `ConeBeamModel.back_project_one_view_to_pixel_batch` (`cone_beam.py:477`) | per-view back kernel; rolled `jax.lax.map` over slice-bands (`:526`) + `jnp.transpose` (`:529`) |
| **cone band-kernel** | `ConeBeamModel.back_project_one_view_to_band` (`cone_beam.py:671`) | per-view back kernel producing one global slice band; internal transpose → `input_transpose_fusion` |
| **cone forward-kernel** | `ConeBeamModel.forward_project_pixel_batch_to_one_view` (`cone_beam.py:275`) | per-view forward; rolled `jax.lax.map` over detector rows (`:470`) |
| **parallel pixel-kernel** | `ParallelBeamModel.back_project_one_view_to_pixel_batch` (`parallel_beam.py:286`) | per-view back; **no band kernel** — sharded path crops detector rows instead |
| **parallel forward-kernel** | `ParallelBeamModel.forward_project_pixel_batch_to_one_view` (`parallel_beam.py:222`) | per-view forward; channel `.at[n,:].add` scatter (`:280`) |
| **pixel driver** | `projectors._sparse_back_project` (`projectors.py:367`) = `projector_functions.sparse_back_project` | jitted scan/map/vmap driver wrapping a pixel-kernel |
| **band driver** | `projectors._sparse_back_project_band` (`projectors.py:403`) = `projector_functions.sparse_back_project_band` | jitted driver wrapping the band-kernel |
| **forward driver** | `projectors._sparse_forward_project` (`projectors.py:332`) = `projector_functions.sparse_forward_project` | jitted forward driver |
| **user back entry** | `TomographyModel.sparse_back_project` (`tomography_model.py:1489`) → `_sparse_back_project_sharded` (`:1618`) | dispatch |
| **GPU n=1 "short-circuit"** | the branch in `_sparse_back_project_sharded` (`tomography_model.py:1685`) → `_sparse_back_project_single_device` (`:1516`) → **pixel driver** | single-GPU recon skips the band path |
| **sharded band path** | `_sparse_back_project_sharded` → `_back_project_all_bands` (`:1758`) → `_back_project_view_shard_to_band` (`:1867`) → **band driver**; reduce-scatter `sum_band_to_owner` (`_sharding/transfer.py`) | multi-device |
| **qGGMRF prior compute** | `qggmrf.qggmrf_gradient_and_hessian_at_indices` (`qggmrf.py:71`) → `qggmrf_grad_and_hessian_per_cylinder` (`:136`) | the per-cylinder elementwise prior |
| **qGGMRF host-halo** | `TomographyModel._extract_halos` (`tomography_model.py:720`) / `qggmrf.extract_halos` (`qggmrf.py:305`); sharded prior `qggmrf.qggmrf_gradient_and_hessian_sharded` (`qggmrf.py:373`) | per-subset neighbor-slice exchange |

---

## Coverage matrix

| op × geometry | CPU | GPU1 | GPUn | notes |
|---|:--:|:--:|:--:|---|
| **back · cone** | ✓ | ✓ | ✓ | the deeply-profiled corner |
| **back · parallel** | — | — | — | gap; `back_project_one_view_to_pixel_batch` (`parallel_beam.py:286`), no band kernel |
| **forward · cone** | ✓ | ~ | — | CPU traced (lax.map-write-bound); GPU1 timing only — it's the *slow* op on GPU (below) |
| **forward · parallel** | — | — | — | gap |
| **qGGMRF prior** (geom-independent) | ? | — | ? | only `lessons.md`-derived hypotheses |

Biggest gaps / next priorities: **cone forward** (GPU-dominant projector cost, un-traced),
**parallel beam** (both projectors, no data), **qGGMRF** (no fresh measurement).

---

## Back projection

### Cone — `✓` CPU/GPU1/GPUn (most complete)

**Bottlenecks**
- `CPU` — in `back_project_one_view_to_band` (`cone_beam.py:671`, the kernel CPU uses): gather-dominated
  (`bitcast_gather_fusion` ≈9.5 s/iter) + cone coordinate math `jnp.arctan2`/`jnp.cos` (`cone_beam.py:727–731`)
  ≈5 s/iter. The alternative `back_project_one_view_to_pixel_batch` (`cone_beam.py:477`) cache-**cliffs**
  ≥~200³ — its rolled `lax.map` (`:526`) + `jnp.transpose` (`:529`) materialize a `views×npix×slices` stack;
  the GPU n=1 short-circuit (`tomography_model.py:1685`) is gated OFF on CPU for exactly this reason. `[✓ trace+ablation]`
- `GPU1` — via the short-circuit → `_sparse_back_project_single_device` → **pixel driver** running
  `back_project_one_view_to_pixel_batch` (`cone_beam.py:477`), 69 ms. Two dominant XLA fusions: the
  per-view accumulate `loop_add_fusion` is **memory-access-pattern-bound** (96% memory-pipe, **8% HBM**,
  29% L2 → saturates on-chip L1/LSU from the gather/scatter, *not* bandwidth); the rolled-`lax.map` write
  `loop_dynamic_update_slice_fusion` (`cone_beam.py:526`) is **compute-bound** (82% SM). occ 82–95%. `[✓ ncu]`
- `GPUn` — sharded band path (`_back_project_all_bands`, `tomography_model.py:1758`) runs
  `back_project_one_view_to_band` (`cone_beam.py:671`). **Non-monotonic**: n=2 (94.7 ms) is 1.3× *slower*
  than n=1 (72.8 ms). Cause: the band kernel's transpose fusions (`input_transpose_fusion`,
  `input_cosine_transpose_fusion`) are **L1/TEX-cache-bound** (99–100% L1, 6–17% HBM, 13–44% SM). The
  reduce-scatter `sum_band_to_owner` (`_sharding/transfer.py`) is cheap (~3.5 ms) — **comms are not the
  limiter, the band kernel's transpose is.** `[✓ trace+ncu]`

**Possible improvements**
- `GPUn` — **restructure `back_project_one_view_to_band` (`cone_beam.py:671`) to avoid the transpose**,
  writing pixel-like via `dynamic_update_slice` (as `back_project_one_view_to_pixel_batch` does), so it
  stops saturating L1 — *without* reintroducing the CPU cliff. Large HBM/compute headroom; this gates
  multi-GPU back scaling. `[measured target — the "B4.5 lever" in lessons.md]`
- `GPU1` — **coalesce the `loop_add_fusion` accumulate's scattered transactions** inside
  `back_project_one_view_to_pixel_batch` (the lever is L1/LSU pressure, not HBM — no bandwidth headroom). `[measured]`
- platform-gated kernel selection (the short-circuit at `tomography_model.py:1685`: CPU→band, GPU n=1→pixel)
  already in place. `[done]`

### Parallel — `—` not investigated
- Per-view kernel `back_project_one_view_to_pixel_batch` (`parallel_beam.py:286`) differs structurally:
  detector-**row crop** (slice r ← row r, no cross-row mixing), **no band kernel**, **no cone coordinate
  math / no cosine transpose** — so the cone band kernel's L1-bound transpose is likely absent. Needs
  trace + ncu. `[gap]`

---

## Forward projection

### Cone — `✓` CPU (traced), `~` GPU1 (timing only)
**Bottlenecks**
- `CPU` (✓ traced, 256³, 16.0 s/iter) — **dominated by `bitcast_dynamic-update-slice_fusion` ≈12.2 s/iter
  (~76%)** = the rolled `jax.lax.map` over detector rows in `forward_project_pixel_batch_to_one_view`
  (`cone_beam.py:470`) — the predicted `lax.map`-write materialization, **CONFIRMED**. Secondary: cone
  coordinate math `cosine_divide_fusion` ≈2.4 s (~15%) + the sinogram `wrapped_scatter` ≈1.0 s (~6%).
  Contrast with back (gather-bound): forward is **lax.map-write-bound** — different dominant op, same
  `lax.map`-rolling root cause as the back pixel-kernel's `lax.map`+transpose. `[✓ trace]`
- `GPU1` — same kernel via the **forward driver** (`projectors._sparse_forward_project`, `projectors.py:332`):
  warm **599 ms = ~8.7× the cone back pixel-kernel (69 ms)** and ~3.3× the band kernel — the **dominant GPU
  projector cost**, still un-traced. Since it's only ~comparable to back on CPU but 8.7× on GPU, the `lax.map`
  at `:470` likely **serializes much worse on GPU**. `[~ timing, exp 3]`

**Possible improvements**
- **Restructure the rolled `jax.lax.map` over detector rows (`cone_beam.py:470`) to avoid the
  `dynamic_update_slice` materialization** — the same lever as the back pixel-kernel's `lax.map`+transpose,
  and it likely helps forward on *both* platforms. `[hypothesis — CPU bottleneck measured]`
- **Next experiment:** trace + ncu forward on GPU1 to confirm the `lax.map` serialization is the 8.7× cause
  (and whether a `vmap`/restructured accumulation removes it).

### Parallel — `—` not investigated
- `forward_project_pixel_batch_to_one_view` (`parallel_beam.py:222`): channel `.at[n,:].add` scatter
  (`:280`), no vertical-fan `lax.map`. Likely a different (simpler) profile than cone. `[gap]`

---

## qGGMRF prior (geometry-independent)

### `?` lessons-derived hypotheses, not re-measured here
**Bottlenecks (from `mbirjax/.claude/lessons.md`, not re-profiled)**
- The **per-subset host-halo exchange** — `TomographyModel._extract_halos` (`tomography_model.py:720`) /
  `qggmrf.extract_halos` (`qggmrf.py:305`) ≈1.35 ms/call — plus per-shard Python dispatch + `assemble_sharded`
  **don't amortize** when the actual prior compute (`qggmrf_gradient_and_hessian_at_indices`, `qggmrf.py:71`)
  is ~2 ms, so the sharded prior (`qggmrf_gradient_and_hessian_sharded`, `qggmrf.py:373`) goes *backwards* at
  fine granularity (0.47× @1024-pixel subsets vs 1.45× @16384). It's only ~8% of per-subset VCD cost
  (`vcd_recon`, `tomography_model.py:2638`), so it drags overall VCD only slightly. `[? lessons]`

**Possible improvements (from lessons)**
- avoid the per-subset **host round-trip** in `_extract_halos`: on-device `move_shard` halo exchange where
  d2d is safe, or fuse the halo read across subsets. `[? hypothesis — matters only if the prior dominates]`

**Gaps:** no fresh trace/ncu of the per-cylinder prior compute (`qggmrf_grad_and_hessian_per_cylinder`,
`qggmrf.py:136`) on CPU or GPU; no single-device characterization.

---

## Cross-cutting (span all ops)

- **Compile time** — `CPU` ~0.25 s/op/shape, size-invariant, XLA-dominated. `GPU` **autotuning-dominated,
  59 ms → 2.4 s, noisy**; trace+lower (the batching nest `sum_/concatenate_function_in_batches` in
  `projectors.py`) stays ~0.1–0.27 s, a smaller share on GPU. Matters for small/many-shape/first-call
  (VCD per-subset, tests). Refactor lever = fewer distinct autotuned kernels. `[✓ exp 3]` Open probe:
  count distinct compiles (`_jit_*._cache_size()`) in a real `vcd_recon` / test run.
- **Tooling caveat (method, not a projector bottleneck)** — `cost_analysis`/`memory_analysis` are the right
  ruler for capacity/FLOPs, the **wrong** ruler for kernel efficiency (ranked the slow CPU pixel-kernel as
  cheaper; "5 GB accessed" couldn't reveal the GPU access-pattern bound). See README. `[✓]`

---

## Open experiments (to fill the matrix)

1. **Cone forward** trace + ncu (CPU & GPU1) of `forward_project_pixel_batch_to_one_view` (`cone_beam.py:275`)
   — top priority; dominates GPU projector time and is un-traced.
2. **Parallel beam** back + forward (`parallel_beam.py:286` / `:222`) — contrast with cone (row-crop, scatter,
   no cone coord math).
3. **qGGMRF** trace + ncu of `qggmrf_grad_and_hessian_per_cylinder` (`qggmrf.py:136`) single-device.
4. **512³ scale-up** of the cone-back picture (inversion depth at production size).
5. `ncu --set full` on the cone-back GPU1 `loop_add_fusion` — name the exact saturated pipe (L1 vs LSU vs atomics).
