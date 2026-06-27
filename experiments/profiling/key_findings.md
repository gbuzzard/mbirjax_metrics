# Key findings — projector & prior bottleneck inventory

A scannable inventory of **bottlenecks** and **possible improvements** for
`{forward, back, qGGMRF}` × `{parallel, cone}`, on CPU and GPU (single- and multi-device).
This is the *inventory*; see [`README.md`](README.md) for methodology, the narrative arc, the
tool-to-question map, and how to run each script.

**Legend** — confidence: `✓` measured here · `~` partial (timing only / single size) · `?` hypothesis
(from `lessons.md`, not re-measured) · `—` not yet investigated.
**Platforms:** `CPU` (Mac, virtual devices) · `GPU1` (single H100) · `GPUn` (multi-H100, sharded).
Numbers are cone, 256³, warm, n=1 unless noted.
**JAX:** measure the PRODUCTION env — **all numbers below are jax/jaxlib 0.10.1** (CPU + GPU re-measured 2026-06-27).
**0.10.2 is EXCLUDED** (a regression — see Cross-cutting). The regression is **backend-AND-kernel-specific**: it
4×-slowed the cone *back band* kernel on CPU and the cone *forward* kernel on GPU, but left GPU back and CPU
forward untouched — so the GPU back findings (platform inversion, n=2 non-monotonicity, band-transpose L1-bound)
are **CONFIRMED on 0.10.1**, while GPU forward dropped 599→148 ms.
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

All cells jax 0.10.1 (production), re-measured 2026-06-27.

| op × geometry | CPU | GPU1 | GPUn | notes |
|---|:--:|:--:|:--:|---|
| **back · cone** | ✓ | ✓ | ✓ | CPU 3.06 s (`multiply`/`reduce-window`, NOT gather); GPU confirmed (inversion 0.38×, n=2 non-monotonic, band L1-bound) |
| **back · parallel** | ✓ | — | — | CPU: 524 ms, `multiply_add`-bound, no cliff |
| **forward · cone** | ✓ | ✓ | — | CPU 16.9 s `lax.map`-write-bound; GPU 148 ms `input_scatter`-bound (DIFFERENT bottleneck per platform) |
| **forward · parallel** | ✓ | — | — | CPU: 1.32 s, `wrapped_scatter`-bound |
| **qGGMRF prior** (geom-independent) | ? | — | ? | only `lessons.md`-derived hypotheses |

Biggest gaps / next priorities: **parallel beam on GPU**, **qGGMRF** (no fresh measurement), **cone forward GPU ncu**
(`input_scatter_fusion` roofline), **512³ scale-up**.

---

## Back projection

### Cone — `✓` CPU/GPU1/GPUn (all 0.10.1; GPU confirmed unchanged by the regression)

**Bottlenecks**
- `CPU` (0.10.1, 256³) — `back_project_one_view_to_band` (`cone_beam.py:671`, the kernel CPU uses):
  **3.06 s/iter (n=1)**, dominated by **`broadcast_multiply_fusion` ≈2.19 s (~72%)** + `wrapped_reduce-window`
  ≈0.29 s; cone coordinate math (`atan2`/`cosine_divide`, `cone_beam.py:727–731`) now small (≈0.03 s). The
  band kernel is **lean** (591 MB temp, 2.5 GB accessed). n=2 = 2.68 s (1.14× — shared-bus ceiling), reduce-scatter
  cheap. The alternative `back_project_one_view_to_pixel_batch` (`cone_beam.py:477`) cache-**cliffs** ≥~200³:
  pixel 23.1 s vs band 2.94 s = **7.86×** (matches lessons.md's ~8×); its rolled `lax.map` (`:526`) +
  `jnp.transpose` (`:529`) materialize a `views×npix×slices` stack, so CPU production uses the band kernel
  (short-circuit gated OFF on CPU, `tomography_model.py:1685`). `[✓ trace+ablation, 0.10.1]`
  ⚠ **0.10.2 artifacts now corrected:** on the excluded 0.10.2 the band kernel was 11.8 s (4× slower),
  `bitcast_gather_fusion`-dominated (~9.5 s), 29.6 GB accessed / 6554 MB temp (~12× inflated), and the cliff
  read only 2.05× (0.10.2 slowed the band too). The "gather-bound" + "2× cliff" + "band does more memory work"
  claims were 0.10.2 regression artifacts.
- `GPU1` (0.10.1, ✓ CONFIRMED — pixel was NOT a regression victim) — via the short-circuit, pixel kernel **69 ms**:
  accumulate `loop_add_fusion` **L1-cache-bound** (96% mem-pipe, **8% HBM**, **97% L1**, 7% SM → on-chip L1/LSU
  from the gather/scatter, not bandwidth); scatter-write `loop_dynamic_update_slice_fusion` compute-bound (82% SM).
  Identical to 0.10.2. `[✓ trace+ncu, 0.10.1]`
- `GPUn` (0.10.1, ✓ CONFIRMED — band held on GPU; the regression was CPU-only for back) — **non-monotonic**:
  n=2 **94.7 ms** > n=1 **72.5 ms** (each of 2 GPUs busy ~91 ms; band 1.3× slower). Band kernel
  `input_transpose_fusion`/`input_cosine_transpose_fusion` **L1/TEX-bound** (99.9% L1, **6% HBM**, 13–44% SM);
  reduce-scatter `sum_band_to_owner` cheap (~3.4 ms). **Comms are not the limiter — the band transpose is.**
  Identical to 0.10.2. `[✓ trace+ncu, 0.10.1]`

**Possible improvements**
- `GPUn` — **restructure `back_project_one_view_to_band` (`cone_beam.py:671`) to avoid the transpose** (write
  pixel-like via `dynamic_update_slice`), so it stops saturating L1 — *without* reintroducing the CPU cliff. Large
  HBM/compute headroom; gates multi-GPU back scaling. **Confirmed real on production jax.** `[measured target — the "B4.5 lever"]`
- `GPU1` — coalesce the `loop_add_fusion` accumulate's scattered L1/LSU transactions (no HBM headroom). `[measured]`
- platform-gated kernel selection (`tomography_model.py:1685`: CPU→band, GPU n=1→pixel) is in place. `[done]`

### Parallel — `✓` CPU (0.10.1)
**Bottlenecks**
- `CPU` (0.10.1, 256³) — `back_project_one_view_to_pixel_batch` (`parallel_beam.py:286`): **524 ms** (≈6× faster
  than cone band's 3.06 s), dominated by **`multiply_add_fusion` ≈0.25 s (~47%)** (the accumulate) + `ynn_fusion`
  ≈0.10 s; `dynamic_update_slice` ≈0 (no `lax.map`). **No cliff, no cone coord math, no transpose** — the
  detector-**row crop** (slice r ← row r) + a clean multiply-add. Confirms the `lax.map` pathologies are cone-only. `[✓ trace]`
- `GPU` — `—` not investigated. `[gap]`

**Possible improvements** — none evident on CPU (already cheap; accumulate-bound).

---

## Forward projection

### Cone — `✓` CPU + GPU1 (0.10.1). **The bottleneck is PLATFORM-DIVERGENT.**
**Bottlenecks** — cone forward is `lax.map`-write-bound on CPU but `scatter`-bound on GPU; the 0.10.2 regression
hit it on GPU (4×) but NOT CPU.
- `CPU` (✓ traced, 256³, 16.9 s/iter, 0.10.1 — unaffected by the regression) — **dominated by
  `bitcast_dynamic-update-slice_fusion` ≈12.2 s/iter (~72%)** = the rolled `jax.lax.map` over detector rows in
  `forward_project_pixel_batch_to_one_view` (`cone_beam.py:470`). Secondary: cone coordinate math
  `cosine_divide_fusion` ≈3.2 s (~19%) + the sinogram `wrapped_scatter` ≈1.0 s (~6%). 13× the parallel forward
  (1.3 s) on the same jax. `[✓ trace]`
- `GPU1` (✓ traced, 256³, **148 ms**, 0.10.1) — **dominated by `input_scatter_fusion` ≈37 ms** (the scatter
  INTO the sinogram), with `loop_dynamic_update_slice_fusion` (the `:470` `lax.map`) ≈12 ms secondary +
  `loop_divide_fusion` ≈8 ms. So on GPU the **sinogram scatter dominates, not the `lax.map`** — opposite of CPU.
  ⚠ **Was a 0.10.2 victim: 599 ms → 148 ms (4×) on 0.10.1.** `[✓ trace]`

**Possible improvements** — the lever is platform-specific:
- `CPU` — restructure the rolled `jax.lax.map` over detector rows (`cone_beam.py:470`) to avoid the
  `dynamic_update_slice` materialization. `[CPU measured]`
- `GPU` — the `input_scatter_fusion` (sinogram scatter) is the target; ncu it next for a roofline. `[GPU measured; ncu pending]`

### Parallel — `✓` CPU (0.10.1)
**Bottlenecks**
- `CPU` (0.10.1, 256³) — `forward_project_pixel_batch_to_one_view` (`parallel_beam.py:222`): **1.32 s**,
  dominated by **`wrapped_scatter` ≈1.04 s (~79%)** = the channel `.at[n,:].add` scatter (`parallel_beam.py:280`)
  + `bitcast_copy_fusion` ≈0.45 s. **Scatter-bound, no `lax.map`** — 13× faster than cone forward, confirming
  the `lax.map`-rolling cost is cone-specific. `[✓ trace]`
- `GPU` — `—` not investigated. `[gap]`

**Possible improvements** — the scatter (`.at[].add`) is the lever if parallel forward ever dominates; cheap today.

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

- **⚠ jax 0.10.2 REGRESSION — backend-AND-kernel-specific; measure the production env.** 0.10.2 (now excluded)
  slowed **different kernels on different backends**: on **CPU** it compiled the cone *back band* kernel
  (`back_project_one_view_to_band`) `bitcast_gather`-heavy, **4× slower (11.8→2.94 s), ~12× memory traffic
  (29.6→2.5 GB)**; on **GPU** it 4×-slowed the cone *forward* kernel (599→148 ms). It left **GPU back** (pixel
  69 ms + band 180 ms — unchanged, ncu-confirmed) and **CPU forward** (16.9 s — unchanged) alone. So you CANNOT
  assume a kernel hit on one backend is hit on the other (and cone forward's bottleneck itself differs by
  backend — CPU `lax.map`, GPU scatter). LESSON: profiling must run the **production env + its pins**; stamp the
  jax version on every result and gate on it (the `run_gpu_all.sh` guard now does). Earlier cone-back CPU
  "findings" (gather-bound; 2×-not-8× cliff; band heavier than pixel) were the 0.10.2 artifact, now corrected.
- **Compile time** (CPU 0.10.1) — ~0.18–0.24 s/op/shape, size-invariant, XLA-dominated; trace+lower (the batching
  nest `sum_/concatenate_function_in_batches` in `projectors.py`) ~0.1 s. (GPU was autotuning-dominated, 59 ms→2.4 s,
  noisy — on 0.10.2, re-confirm.) Matters for small/many-shape/first-call (VCD per-subset, tests). Refactor lever =
  fewer distinct autotuned kernels. `[✓ exp 3]` Open probe: count distinct compiles (`_jit_*._cache_size()`) in a real `vcd_recon`/test run.
- **Tooling caveat (method)** — `cost_analysis`/`memory_analysis` are the right ruler for capacity/FLOPs, the
  **wrong** ruler for the cache cliff: on 0.10.1 the slow pixel kernel has *fewer* FLOPs (8.7 vs 15.8 G) and only
  2× the bytes (5.0 vs 2.5 GB) yet is **8× slower** — the `lax.map`+transpose cache-locality cost is invisible to
  logical FLOP/byte counts. (The stronger "counters point the *wrong* way" was itself a 0.10.2 artifact.) See README. `[✓]`

---

## Open experiments (to fill the matrix)

~~0. RE-MEASURE ALL GPU on 0.10.1~~ — **DONE 2026-06-27** (`run_gpu_all.sh`); GPU back confirmed, GPU forward
   corrected (4× faster). Cone back+forward rows are now ✓ on 0.10.1.
1. **Cone forward GPU ncu** — roofline the `input_scatter_fusion` (the GPU bottleneck) of
   `forward_project_pixel_batch_to_one_view` (`cone_beam.py:275`); update `ncu_back_projection.py`-style regex.
2. **Parallel beam on GPU** back + forward (`parallel_beam.py:286` / `:222`) — CPU done; GPU is the gap.
3. **qGGMRF** trace + ncu of `qggmrf_grad_and_hessian_per_cylinder` (`qggmrf.py:136`) single-device.
4. **512³ scale-up** of the cone-back picture (inversion depth at production size, on 0.10.1).
5. `ncu --set full` on the cone-back GPU1 `loop_add_fusion` — name the exact saturated pipe (L1 vs LSU vs atomics).
6. **Faster ncu:** bracket the profiled region with `cudaProfilerStart/Stop` + `ncu --profile-from-start off`
   (the 8-min runs were mostly ncu instrumenting JAX's compile-time autotuning during warmup).
