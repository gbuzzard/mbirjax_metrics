"""
experiments/sharding/scaling_tests/performance_tracking.py
──────────────────────────────────────────────────────────
Nightly / manual REGRESSION engine.  Sweeps GEOMETRY × OP × size × device-count over the
existing ``scaling_common`` harness, measuring time + peak memory per cell (a tolerant
correctness fingerprint and the diff/gate come in later phases), and writes ONE dated YAML.

Full design + rationale: ``experiments/sharding/plans/performance_tracking_plan.md``.

This is a THIN driver over ``scaling_common.py`` — do NOT rebuild the measurement machinery.
It reuses the isolated-subprocess discipline, ``time_op`` warmup/trials, ``peak_memory_mb``,
the device-count OOM-descent (``run_measure_loop``), throttle sampling, the path/band YAML
fields, and YAML I/O.  The genuinely new pieces over ``cone_baseline_scaling.py`` are:

  * **GEOMETRY is a sweep dimension** (not a module constant), so a ``Config`` object drives the
    run.  A fresh worker subprocess cannot see the orchestrator's ``Config``, so the orchestrator
    serializes it to a temp YAML and passes ``--config`` to each worker (sweep coordinates go on
    argv for readability).  This is strictly more robust than cone_baseline's module-global read
    and is what lets one engine serve the nightly and the manual launcher.
  * **inline mode** (``Config.inline`` / ``--inline``) runs the worker body IN-PROCESS — no
    subprocess hop, fully step-through-able in PyCharm.  The cost: peak memory is then cumulative
    across the sweep, not per-config (``peak_bytes_in_use`` is a high-water mark), so inline is
    for debugging logic/correctness; trust the isolated-subprocess numbers for the memory ruler.
  * **per-device-count isolation** (isolated/subprocess path only): the orchestrator spawns a
    SEPARATE worker for each device count of a cell (descending, with the OOM early-stop lifted up
    from ``run_measure_loop``), because ``peak_bytes_in_use`` is a process-cumulative high-water
    mark with no in-process reset in jax — a shared-process descent lets one count's warmup/compile
    scratch inflate the next count's peak (the n>1 memory-gate false-positive source).

Roles (mirrors cone_baseline_scaling.py):
  - orchestrator (default, no args)            : ``run(Config)`` — per (geom, op, size, device
                                                  count) spawn a worker (or, inline, one call for
                                                  the whole cell), collect rows, write YAML.
  - worker --mode setup                        : report platform / devices.
  - worker --mode measure --config --geometry --op --size --device-counts : measure the given
                                                  device count(s) of one cell and write its rows
                                                  (the isolated orchestrator passes ONE count).

mbirjax/jax are imported INSIDE the worker functions only (device-setup-first; the default
orchestrator role stays JAX-free so a subprocess worker can read peak memory cleanly).  In
``--inline`` mode the orchestrator DOES import them (the documented tradeoff above).
"""
import os
import sys
import gc
import io
import re
import contextlib
import argparse
import datetime
import tempfile
import dataclasses
from dataclasses import dataclass, field
from collections import OrderedDict

import scaling_common as sc

import numpy as np


# ── Run configuration ─────────────────────────────────────────────────────────
# The Config defaults encode the NIGHTLY sweep; the manual launcher and main()
# override a subset.  See the plan for the field-by-field rationale.  A worker
# reconstructs this from the temp YAML the orchestrator writes (from_dict tolerates
# extra/missing keys so the schema can evolve without breaking serialized configs).
@dataclass
class Config:
    # sweep dimensions
    geometries: list = field(default_factory=lambda: ["parallel", "cone", "translation", "multiaxis_parallel", "denoiser"])
    ops: list = field(default_factory=lambda: ["direct_filter", "forward", "back", "vcd_nonconst"])
    # Per-geometry op OVERRIDES (else `ops` is used).  translation/multiaxis only have geometry-SPECIFIC
    # compute in the projectors + filter; their vcd recon is the qGGMRF outer loop shared with
    # parallel/cone (tracked there), so measuring it would be redundant — drop vcd for those two.
    # denoiser: its ONLY real op is `denoise` (the qGGMRF VCD outer loop with identity projectors —
    # the vcd_nonconst analog); forward/back are the identity and there is no FBP, so nothing else.
    geom_ops: dict = field(default_factory=lambda: {
        "translation": ["direct_filter", "forward", "back"],
        "multiaxis_parallel": ["direct_filter", "forward", "back"],
        "denoiser": ["denoise"],
    })
    device_counts: list = field(default_factory=lambda: [1, 2, 4])
    # SINOGRAM sizes (n_views, n_rows, n_channels) — ASYMMETRIC (all three differ) to surface
    # axis swaps; one DIVIDING + one NON-DIVIDING (all-odd) per platform to exercise padding;
    # plus a GPU 1024-class capacity size.  The recon shape is auto-derived per geometry.
    # The LARGEST CPU size of each geometry is ALSO measured on GPU (the first GPU entry below) so the
    # dashboard's CPU<->GPU cross-platform correctness check has a shared cell to compare — cheap on GPU,
    # no extra CPU cost (CPU already runs it).  See the correctness-gating design note D7.
    sizes: dict = field(default_factory=lambda: {
        "cpu": [(128, 112, 96), (129, 113, 97), (200, 208, 160)],
        "gpu": [(200, 208, 160), (512, 448, 384), (513, 449, 385), (1024, 1008, 992)],  # 200³ = shared w/ CPU
    })
    # Per-geometry size OVERRIDES (else `sizes[plat]` is used).  The new translation/multiaxis
    # geometries are sharding-baseline work: keep them SMALL (top out at 512 on GPU).  Translation's
    # tuple is (n_views=15 fixed, n_det_rows, n_det_channels); make_model derives its recon from these.
    # The first GPU entry mirrors the largest CPU size (the CPU<->GPU cross-platform shared cell, D7).
    geom_sizes: dict = field(default_factory=lambda: {
        "multiaxis_parallel": {
            "cpu": [(96, 80, 64), (128, 112, 96), (129, 113, 97)],
            "gpu": [(129, 113, 97), (256, 224, 192), (512, 448, 384), (513, 449, 385)],  # 129³ = shared w/ CPU
        },
        "translation": {
            "cpu": [(15, 64, 64), (15, 65, 65)],
            "gpu": [(15, 65, 65), (15, 256, 256), (15, 257, 257)],  # 15x65x65 = shared w/ CPU
        },
        # denoiser: the size tuple IS the IMAGE shape (rows, cols, slices) — no sinogram.  Sized from the
        # measured envelope (GPU: 1024³-class fits, 1600³ OOMs on n=1/2/4; CPU ~256³-class is a few s).
        # Asymmetric to surface axis swaps; the all-odd cell forces slice-axis padding; (1024,1008,992) is
        # the GPU capacity probe (single-trial — see single_trial_sizes).  The largest CPU cell is mirrored
        # to GPU as the cross-platform shared cell.
        "denoiser": {
            "cpu": [(128, 144, 160), (225, 241, 257)],
            "gpu": [(225, 241, 257), (512, 448, 384), (1024, 1008, 992)],  # 225x241x257 = shared w/ CPU
        },
    })
    # Sizes where every op runs trials=1 (capacity/memory check, not a timing ruler).
    single_trial_sizes: list = field(default_factory=lambda: ["1024x1008x992"])

    # vcd (not yet wired up)
    vcd_iterations: int = 3
    weight_mode: str = "nonconstant"
    weight_seed: int = 13

    # denoiser (QGGMRFDenoiser.denoise — the vcd analog with identity projectors).  A FIXED sigma + EXACT
    # iteration count (stop_threshold_change_pct=0) + seeded input image make the fingerprint
    # deterministic (so the cross-device / cross-platform / vs-main checks are meaningful).
    denoise_iterations: int = 20   # realistic (the demo's max; matches the measured timing envelope) and
                                   # surfaces sharding-communication overhead better than a 3-iter ruler
    denoise_sigma: float = 0.1
    denoise_sharpness: float = 0.0

    # measurement
    warmup: int = 1
    trials_by_op: dict = field(default_factory=lambda: {
        "direct_filter": 3, "forward": 3, "back": 3, "vcd_nonconst": 1, "denoise": 1})
    inline: bool = False   # True = single-process, debuggable (memory not per-config)

    # geometry / seeds
    cone_sdd_over_channels: float = 4.0
    multiaxis_elevation_deg: float = 25.0   # out-of-plane tilt for MultiAxisParallelModel (demo value)
    # Translation/TCT fixed geometry (from experiments/translation/TCT_generate_phantom.py, exp1).
    # The detector (n_rows, n_channels) from the size tuple is the knob; n_views is fixed at num_x*num_z.
    tct_sdd: float = 190000.0    # source->detector (ALU) — exp1 geometry (well-conditioned cone angle)
    tct_sid: float = 70000.0     # source->iso (ALU); magnification = sdd/sid ~= 2.71
    tct_delta_det: float = 75.0  # detector pixel pitch (ALU)
    tct_num_x: int = 5           # x translations
    tct_num_z: int = 3           # z translations  -> n_views = 15
    tct_recon_rows: int = 40     # recon rows (axis perpendicular to detector) — fixed; cols/slices derived
    input_seed: int = 0
    measure_seed: int = 7

    # io / provenance
    out_dir: str = ""      # stable nightly dir, or results/manual/<tag> (required at run time)
    date: str = ""         # stamped by the orchestrator (never datetime.now() in a worker)
    run_tag: str = ""
    # Dependency-canary provenance (tooling/regression/dependency_canary_plan.md).  dep_gen = the identity
    # of the installed dependency set (monotonic per platform); when >0 the run filename gets a `_gNNNN`
    # suffix so multiple runs of ONE commit with different deps don't collide (gen 0 = the historical
    # name, no suffix).  run_reason records what triggered this run.  Default = a plain code run.
    dep_gen: int = 0
    run_reason: str = "commit"   # "commit" | "jax-step" | "code-step" | "deps-step"
    jax_available: str = ""      # PyPI-latest jax at run time (REG_JAX_AVAILABLE from the wrapper).  Records
                                 # what was AVAILABLE even when the install resolved to an older jax (held
                                 # back by the env Python floor or the pyproject jax!=... exclusion), so the
                                 # dashboard can say "X available, installed Y" instead of a bare "jax Y".
    lib_root: str = ""     # library checkout to MEASURE (PYTHONPATH + provenance); "" -> beta_root()
                           # (this harness's own checkout).  The nightly sets it to a per-branch worktree.

    # diff / gate
    gate: bool = True               # set the process exit code on a HARD regression
    compare_to_prior: bool = True   # compare against the most-recent prior dated file in out_dir
                                    # (this branch's own previous commit) — the sole gate reference.
                                    # Cross-branch comparison (vs main/prerelease) + best-ever drift
                                    # are surfaced on the dashboard, not gated here.
    mem_hard_pct: float = 8.0       # memory growth threshold (%); HARD on GPU, soft on CPU
    speedup_warn_pct: float = 15.0  # speedup-ratio drop WARN threshold (%); soft on all platforms
    time_soft_pct: float = 25.0     # absolute-time WARN threshold (%)
    fp_rtol_single: float = 1e-5    # fingerprint robust-aggregate rel tol (single-shot ops)
    fp_rtol_iter: float = 1e-4      # ... for the iterated vcd
    k_sample_tol: int = 1           # allowed deviating fingerprint samples before a soft flag

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


# Cone recon_shape pins — padding-policy decoupling ────────────────────────────
# The cone auto recon_shape depends on the library's axial-padding policy, so a change to
# that policy would silently move recon_shape (and thus every cone cell's peak_bytes /
# wall-time baseline).  We pin cone recon_shape per sinogram size to the current shipped
# shape so padding-policy changes no longer move the vs-prior time/memory series — that
# regime is tracked deliberately via the run-header policy block and results/annotations.yaml
# instead.  The pinned values equal the current auto-derived shapes (baselines do not move);
# pinning only makes _recon_geometry_is_auto False so future auto-padding can't reshape a cell.
CONE_RECON_SHAPE_PINS = {
    (128, 112, 96):    (96, 96, 126),
    (129, 113, 97):    (97, 97, 129),
    (200, 208, 160):   (160, 160, 234),
    (512, 448, 384):   (384, 384, 504),
    (513, 449, 385):   (385, 385, 507),
    (1024, 1008, 992): (992, 992, 1134),
}


# ── Op-specific builders (used by the worker body) ────────────────────────────
def make_model(config, geometry, size):
    """Build a single-device model of ``geometry`` for SINOGRAM ``size``.

    ``size`` = (n_views, n_rows, n_channels).  The recon shape is auto-derived by the model
    (for cone it differs from the sinogram shape).  Representative cone geometry: magnification
    2 (source_detector_dist = cone_sdd_over_channels * channels, source_iso_dist = half that),
    matching the test-suite convention.

    translation: n_views is FIXED at num_x*num_z (the size tuple's view count is ignored); the
    detector (n_rows, n_channels) is the knob.  We scale the translation spacings with the detector
    (at a fixed pixel pitch) so calc_tct_recon_params yields recon cols ~ n_channels, slices ~ n_rows,
    with recon rows pinned to tct_recon_rows (the few-rows axis perpendicular to the detector).
    multiaxis_parallel: like parallel but angles are (azimuth, elevation); recon matched to parallel.
    """
    import mbirjax
    n_views, n_rows, n_channels = size
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    if geometry == "parallel":
        model = mbirjax.ParallelBeamModel((n_views, n_rows, n_channels), angles)
    elif geometry == "cone":
        sdd = config.cone_sdd_over_channels * n_channels
        sid = sdd / 2.0
        model = mbirjax.ConeBeamModel((n_views, n_rows, n_channels), angles,
                                      source_detector_dist=sdd, source_iso_dist=sid)
        # Pin recon_shape so the cone axial-padding POLICY (which is changing) cannot move
        # the measured shape/memory/time — see CONE_RECON_SHAPE_PINS above.
        pin = CONE_RECON_SHAPE_PINS.get((int(n_views), int(n_rows), int(n_channels)))
        if pin is not None:
            model.set_params(recon_shape=tuple(int(x) for x in pin))
        else:
            # A new cone size with no pin falls back to the library auto shape (current
            # behavior) but is flagged so its baseline can be pinned deliberately.
            print(f"WARNING: cone size {(n_views, n_rows, n_channels)} has no recon_shape "
                  f"pin; using auto {tuple(int(x) for x in model.get_params('recon_shape'))}"
                  f" — add it to CONE_RECON_SHAPE_PINS to decouple from padding policy.",
                  file=sys.stderr)
    elif geometry == "multiaxis_parallel":
        azimuths = np.linspace(0, np.pi, n_views, endpoint=False)
        elevation = np.full(n_views, np.deg2rad(config.multiaxis_elevation_deg), dtype=np.float64)
        angles2 = np.column_stack([azimuths, elevation])
        model = mbirjax.MultiAxisParallelModel((n_views, n_rows, n_channels), angles2)
        # Match the parallel recon convention so the problem size is comparable to ParallelBeamModel.
        pb = mbirjax.ParallelBeamModel((n_views, n_rows, n_channels), azimuths)
        model.set_params(recon_shape=pb.get_params('recon_shape'))
    elif geometry == "translation":
        sdd, sid, delta_det = config.tct_sdd, config.tct_sid, config.tct_delta_det
        num_x, num_z = config.tct_num_x, config.tct_num_z
        delta_voxel = delta_det / (sdd / sid)   # isotropic in-plane voxel pitch at iso
        # The recon (rows, cols, slices) is set DIRECTLY: rows fixed at tct_recon_rows, and cols<-channels,
        # slices<-rows so it scales with the sino detector (the user's mapping).  ISOTROPIC voxels
        # (aspect=1) keep the 40 rows well within the source (sid >> 40*delta_voxel).  The translation
        # spacings span the recon FOV so the views actually move across the object.  We do NOT use
        # calc_tct_recon_params here: its anisotropic TCT row pitch pushes 40 rows past a small source.
        cols, slices = n_channels, n_rows
        x_spacing = cols * delta_voxel / (num_x - 1)
        z_spacing = slices * delta_voxel / (num_z - 1)
        tv = mbirjax.gen_translation_vectors(num_x, num_z, x_spacing, z_spacing)
        sino_shape = (int(tv.shape[0]), n_rows, n_channels)
        model = mbirjax.TranslationModel(sino_shape, tv, source_detector_dist=sdd, source_iso_dist=sid)
        model.set_params(delta_det_channel=delta_det, delta_det_row=delta_det)
        model.set_params(delta_voxel=delta_voxel, voxel_row_aspect=1.0, voxel_slice_aspect=1.0)
        # Setting recon_shape explicitly also decouples translation from the padding policy
        # (_recon_geometry_is_auto is already False here) — same rationale as CONE_RECON_SHAPE_PINS.
        model.set_params(recon_shape=(config.tct_recon_rows, cols, slices))
    elif geometry == "denoiser":
        # The denoiser has no sinogram/projection: the size tuple IS the image (= recon) shape, and the
        # model is a QGGMRFDenoiser over it.  n_views/angles above are computed but unused here.
        model = mbirjax.QGGMRFDenoiser(tuple(int(x) for x in size))
        model.set_params(sharpness=config.denoise_sharpness)
    else:
        raise ValueError(f"unknown geometry {geometry!r} "
                         f"(expected parallel/cone/translation/multiaxis_parallel/denoiser)")
    model.set_params(verbose=0)
    return model


def make_indices(model):
    """Full field-of-view pixel indices for the model (deterministic per size)."""
    import mbirjax
    recon_shape = model.get_params('recon_shape')
    return mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))


def make_cylinders(num_pixels, num_slices, seed):
    """Deterministic random recon cylinders (num_pixels, num_slices) float32."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_pixels, num_slices), dtype=np.float32)


def make_sinogram(config, size):
    """Deterministic random sinogram of SINOGRAM ``size`` (numpy float32).

    Projection is linear, so a random sinogram is a valid timing/memory input.
    """
    rng = np.random.default_rng(config.input_seed)
    return rng.random(size, dtype=np.float32)


def make_noisy_image(config, size):
    """Deterministic noisy 3D image (numpy float32) of IMAGE shape ``size`` — the denoiser's input.

    A seeded random image is a valid timing/memory/fingerprint input: with a FIXED ``sigma_noise`` and an
    EXACT iteration count (``stop_threshold_change_pct=0``) the denoise is deterministic, so the output
    fingerprint is reproducible across runs, device counts, and platforms.
    """
    rng = np.random.default_rng(config.input_seed)
    return rng.standard_normal(size, dtype=np.float32)


def run_denoise(model, image, config):
    """Timed op: the qGGMRF MAP denoiser (the vcd analog — same VCD outer loop, identity projectors).

    Passes the HOST image and lets ``denoise`` place/shard it internally — matching ``run_vcd`` (the
    iterative recon's input placement is part of its measured cost, unlike the single-shot ops which
    pre-place via ``to_device``).  ``output_sharded=True`` keeps the result in the device (sharded) form
    — measure the denoise, not the final gather.  Returns just the image (drops the recon dict).

    The denoiser builds its VCD partitions with ``np.random``, so we seed the global RNG first (the
    library's own test does the same): otherwise per-call partition variation (~1e-4) swamps the ~1e-7
    float floor — it would confound the cross-device check (sharded vs n=1) AND make the fingerprint
    irreproducible across runs/platforms.  Seeding makes the result deterministic AND sharding-invariant.

    ``output_sharded`` was added with the sharded denoiser, so it does NOT exist on pre-sharding branches
    (main / prerelease).  Mirror ``run_filter``: try it, and on the resulting ``TypeError`` fall back to
    the plain call — those branches are single-device (n=1) anyway, where the result is the real shape.
    """
    np.random.seed(config.measure_seed)
    kw = dict(sigma_noise=config.denoise_sigma, max_iterations=config.denoise_iterations,
              stop_threshold_change_pct=0.0, print_logs=False)
    try:
        out, _ = model.denoise(image, output_sharded=True, **kw)
    except TypeError:
        out, _ = model.denoise(image, **kw)
    return out


def to_device(model, arr, kind):
    """Pre-place a HOST input on the model's device form, OUTSIDE the timing loop.

    The timed op must measure compute, not the host->device transfer + scatter a numpy input
    incurs on every call ("measure the op, not the scatter").  ``kind`` is 'sino' (view-sharded)
    or 'recon' (slice-sharded).  Falls back to a single-device device_put on pre-sharding code.
    Blocks so the transfer is complete before timing begins.
    """
    import jax
    if kind == "sino" and hasattr(model, "_shard_sinogram"):
        placed = model._shard_sinogram(arr)
    elif kind == "recon" and hasattr(model, "_shard_recon"):
        placed = model._shard_recon(arr)
    else:
        placed = jax.device_put(arr, model.main_device)
    jax.block_until_ready(placed)
    return placed


def run_filter(model, sino):
    """Timed op: the FBP/FDK filter, kept in the device (sharded) form.

    Calls ``model.direct_filter`` — but if THIS geometry didn't override it (an inherited
    ``TomographyModel`` no-op STUB that just warns; e.g. a ``MultiAxisParallelModel`` exposing only
    ``fbp_filter``, on library versions predating its ``direct_filter`` alias), call the geometry's
    REAL filter (``fbp_filter`` / ``fdk_filter``) so we measure the actual filter on every branch
    rather than recording a meaningless no-op.  ``output_sharded=True`` measures the FILTER, not a
    full-sinogram gather at exit; the plain-call fallback handles code predating that kwarg.
    """
    import mbirjax
    fn = getattr(model, "direct_filter", None)
    base = getattr(mbirjax.TomographyModel, "direct_filter", None)
    if fn is None or (base is not None and getattr(type(model), "direct_filter", None) is base):
        fn = getattr(model, "fbp_filter", None) or getattr(model, "fdk_filter", None) or fn
    try:
        return fn(sino, output_sharded=True)
    except TypeError:
        return fn(sino)


def run_forward(model, cylinders, pixel_indices):
    """Timed op: forward projection."""
    return model.sparse_forward_project(cylinders, pixel_indices)


def run_back(model, sino, pixel_indices):
    """Timed op: back projection."""
    return model.sparse_back_project(sino, pixel_indices)


def make_weights(config, size):
    """Deterministic NONCONSTANT weights (positive) for the weighted VCD path.

    All-ones weights skip the weighted gradient/Hessian path; a seeded uniform draw in
    [0.5, 1.5] exercises it while staying positive and reproducible.
    """
    rng = np.random.default_rng(config.weight_seed)
    return rng.uniform(0.5, 1.5, size=size).astype(np.float32)


def build_partitions(model, sino_np, weights, max_iterations, seed=None):
    """Build the VCD pixel partitions + sequence once (device-independent, outside timing).

    initialize_recon constructs the partitions (consuming the global RNG) and the partition
    sequence; we keep only those two.  ``seed`` pins the partition grouping: gen_pixel_partition
    draws from the UN-seeded global RNG, so without this the partitions — and therefore the VCD
    recon — vary run to run (verified ~4e-2), which would make the day-over-day VCD fingerprint
    false-positive.  Seeding here makes VCD reproducible across runs.
    """
    if seed is not None:
        np.random.seed(seed)
    (_sino, _weights, _init, partitions, partition_sequence,
     _granularity, _reg) = model.initialize_recon(
        sino_np, weights=weights, max_iterations=max_iterations, print_logs=False)
    return partitions, partition_sequence


def run_vcd(model, sino_np, weights, partitions, partition_sequence, measure_seed):
    """Timed op: one full VCD reconstruction with NONCONSTANT weights.

    Seeds the global RNG so the subset order is identical on every call (stable timing).
    ``init_recon=None`` lets vcd_recon compute its own direct_recon init (part of the real
    per-recon cost).  Returns only the recon (the fingerprint/correctness target).
    """
    np.random.seed(measure_seed)
    recon, _stats = model.vcd_recon(
        sino_np, partitions, partition_sequence,
        stop_threshold_change_pct=0.0, weights=weights, init_recon=None)
    return recon


# ── Correctness fingerprint ───────────────────────────────────────────────────
def _crop_to_true_shape(arr, true_shape):
    """Crop a possibly-padded device-form output to the TRUE shape and check the padding.

    At a non-dividing count an op may return the padded device form (e.g. 49->50 views,
    41->42 slices).  The fingerprint must be on the TRUE shape so it is comparable across
    device counts and runs.  Returns ``(cropped, padding_zero)`` where padding_zero is:
      - None  if arr is not padded (shape already == true_shape),
      - True/False whether the padded OVERHANG is exactly 0 (a constructed-zero invariant; a
        non-zero overhang is a real padding-leak bug, surfaced rather than hidden).
    """
    arr = np.asarray(arr)
    true_shape = tuple(int(s) for s in true_shape)
    if arr.shape == true_shape:
        return arr, None
    padding_zero = True
    for ax, (a, t) in enumerate(zip(arr.shape, true_shape)):
        if a > t:   # overhang along this axis must be exactly zero
            overhang = arr.take(range(t, a), axis=ax)
            if not bool(np.all(overhang == 0.0)):
                padding_zero = False
    cropped = arr[tuple(slice(0, t) for t in true_shape)]
    return cropped, padding_zero


def fingerprint(result, true_shape, k_samples=12):
    """Tolerant correctness fingerprint of an op output, computed on the TRUE shape.

    Reductions {sum, mean, l2norm} are accumulated in float64 so the fingerprint reflects the
    array's value, not float32 accumulation order (which varies with device count).  ``samples``
    are the exact values at K evenly-spaced, deterministic flat indices.  ``shape``/``dtype`` are
    the exact (structural) part of the gate.  See _crop_to_true_shape for the padding handling.
    """
    cropped, padding_zero = _crop_to_true_shape(result, true_shape)
    flat = np.asarray(cropped).ravel()
    n = int(flat.size)
    flat64 = flat.astype(np.float64)
    idx = (np.linspace(0, n - 1, min(k_samples, n)).astype(int) if n else np.array([], int))
    return {
        "sum": float(flat64.sum()),
        "mean": float(flat64.mean()) if n else 0.0,
        "l2norm": float(np.sqrt(np.sum(flat64 * flat64))),
        "min": float(flat.min()) if n else 0.0,
        "max": float(flat.max()) if n else 0.0,
        "samples": [float(flat[i]) for i in idx],
        "shape": list(np.asarray(cropped).shape),
        "dtype": str(np.asarray(result).dtype),
        "padding_zero": padding_zero,
    }


def path_info(model, op, devs, num_pixels, num_slices):
    """Record WHICH code path this measurement used, so the YAML is self-documenting.

    ``is_sharded``: True = placement/banded path (a 1-device sharded run is a trivial mesh,
    NOT the legacy single-device path).  For the back op also record the sharded band length /
    band count (best-effort), which drive back memory and the horizontal-recompute cost.
    """
    info = {"is_sharded": bool(getattr(model, "is_sharded", False)) or hasattr(model, "shard_devices"),
            "n_shard_devices": len(getattr(model, "shard_devices", None) or devs),
            "platform": devs[0].platform}
    if op == "back":
        try:
            slices_per_dev = num_slices // len(devs)
            fixed = getattr(model, "back_project_slice_band", None)
            band_len = model._slice_band_length(slices_per_dev, len(devs), num_pixels,
                                                fixed_band=fixed)
            bounds = model._balanced_slice_bounds(slices_per_dev, band_len)
            info["back_band_len"] = int(band_len)
            info["back_n_bands_per_shard"] = len(bounds)
        except Exception:   # internal API differs (e.g. legacy code) -> record None
            info["back_band_len"] = None
            info["back_n_bands_per_shard"] = None
    return info


def parse_size_label(label):
    """'128x112x96' -> (128, 112, 96)."""
    return tuple(int(x) for x in label.split("x"))


# ── Worker body (callable inline OR in an isolated subprocess) ────────────────
def measure_cell_group(config, geometry, op, size_label, device_counts, out_file):
    """Measure one (geometry, op, size) across ``device_counts`` (descending, OOM-aware).

    Builds the model + op input once for this size on a single-device base model (host arrays
    the per-count models re-place at entry), then for each device count PINS the model to
    exactly that count, pre-places inputs on the device form, and times only the op.  Cone slice
    padding is not yet implemented, so cone at a non-dividing count is NOT skipped — the op runs
    and any failure is captured as a failure cell (see build_and_time).

    ``out_file`` is used by ``run_measure_loop`` for incremental partial publishing; the caller
    (worker entry, or the inline orchestrator) is responsible for it.  Returns the result dict.
    """
    import mbirjax  # noqa: F401  (device-setup side effect; must precede jax init)
    size = parse_size_label(size_label)
    # The denoiser has no sinogram/projection: its input is a noisy IMAGE of the recon shape, and
    # denoise() builds its own partitions internally — so the sinogram / indices / cylinders / VCD-
    # partition setup below is projection-only.
    is_denoiser = (geometry == "denoiser")
    sino_np = None if is_denoiser else make_sinogram(config, size)
    image_np = make_noisy_image(config, size) if is_denoiser else None

    # Build the model + op input once (device-independent host arrays).  Pin the base model to
    # ONE device so derived inputs carry no multi-device placement; build_and_time configures
    # the real count per measurement.
    base_model = make_model(config, geometry, size)
    if hasattr(base_model, "configure_devices"):   # absent on pre-sharding code
        base_model.configure_devices(1)
    recon_shape = tuple(int(x) for x in base_model.get_params('recon_shape'))
    if is_denoiser:
        idx = cylinders = num_pixels = None
        num_slices = recon_shape[2]
    else:
        idx = make_indices(base_model)
        num_pixels = len(idx)
        num_slices = recon_shape[2]
        cylinders = (make_cylinders(num_pixels, num_slices, config.input_seed)
                     if op == "forward" else None)
    # VCD inputs (built once, device-independent): nonconstant weights + the pixel partitions.
    weights = partitions = partition_sequence = None
    if op == "vcd_nonconst":
        weights = make_weights(config, size)
        partitions, partition_sequence = build_partitions(
            base_model, sino_np, weights, config.vcd_iterations, seed=config.measure_seed)
    del base_model
    gc.collect()

    # TRUE (unpadded) output shape per op, for the fingerprint crop: filter/forward emit the
    # sinogram shape; back emits (num_pixels, num_slices); vcd + denoise emit the recon (image) shape.
    op_true_shape = {
        "direct_filter": tuple(size),
        "forward": tuple(size),
        "back": (num_pixels, num_slices),
        "vcd_nonconst": tuple(recon_shape),
        "denoise": tuple(recon_shape),
    }.get(op, tuple(size))

    trials = 1 if size_label in config.single_trial_sizes else config.trials_by_op.get(op, 3)
    path_by_n = {}
    fp_by_n = {}
    skips = []

    def build_and_time(n, devs):
        # We do NOT pre-skip any (size, count) we expect to OOM (Greg's call): the run_measure_loop
        # descent already handles it — an OOM at device count n stops the descent, skipping the
        # smaller counts (which need MORE per-device memory and would also OOM).  So an OOM is just
        # recorded as a failure cell; nightly runs let it go until done.
        model = make_model(config, geometry, size)
        # Pin EXACTLY these n devices, so the model runs on the same devices peak_memory_mb(devs)
        # reads.  Without this the model auto-shards across ALL devices at construction.
        if hasattr(model, "configure_devices"):
            model.configure_devices(devs)
        # Cone slice padding is not yet implemented, so a non-dividing count for CONE is NOT
        # skipped — we let the op run so the harness records ground truth (Greg's call: validate
        # the failure-capture path + track the eventual fix as a datable failure->success
        # transition).  Empirically (this session, padded cone): `forward` RAISES (run_measure_loop
        # captures it as a failure and continues the descent, since it is not an OOM), while `back`
        # and `direct_filter` already tolerate padding and return the padded DEVICE form (e.g.
        # 49->50 views, 41->42 slices).  No allowlist: the gate fires only on a CHANGE in cell
        # status vs the prior run, so a persistent known failure stays a visible wart
        # without alarming, and the fix surfaces as a fail->ok improvement.
        path_by_n[n] = path_info(model, op, devs, num_pixels, num_slices)
        # Pre-place big host inputs on this config's device form OUTSIDE the timing loop.
        if op == "direct_filter":
            sino_dev = to_device(model, sino_np, "sino")
            run_fn = lambda: run_filter(model, sino_dev)
        elif op == "forward":
            cyl_dev = to_device(model, cylinders, "recon")
            run_fn = lambda: run_forward(model, cyl_dev, idx)
        elif op == "back":
            sino_dev = to_device(model, sino_np, "sino")
            run_fn = lambda: run_back(model, sino_dev, idx)
        elif op == "vcd_nonconst":
            model.setup_logger(print_logs=False)
            run_fn = lambda: run_vcd(model, sino_np, weights, partitions,
                                     partition_sequence, config.measure_seed)
        elif op == "denoise":
            # Pass the HOST image; denoise places/shards it internally (like run_vcd, not to_device).
            run_fn = lambda: run_denoise(model, image_np, config)
        else:
            raise ValueError(f"op {op!r} not implemented")
        stats, result = sc.time_op(run_fn, config.warmup, trials)
        mem_mb, mem_kind = sc.peak_memory_mb(devs)
        # Correctness fingerprint AFTER the memory read (the host gather must not inflate the
        # device peak), on the TRUE shape (crop + padding-zero check; see fingerprint()).
        fp_by_n[n] = fingerprint(result, op_true_shape)
        return stats, mem_mb, mem_kind

    rows, failures = sc.run_measure_loop(
        size_label, device_counts, out_file, build_and_time,
        header_extra=f" | {geometry} | op={op} | recon={recon_shape}",
        print_traceback=False)   # expected failures (e.g. cone padding) -> clean one-liner
    # Stamp each row with its sweep coordinates + the auto-derived recon shape + code-path info.
    for r in rows:
        r["geometry"] = geometry
        r["op"] = op
        r["size"] = size_label
        r["recon_shape"] = list(recon_shape)
        r["trials"] = trials
        r.update(path_by_n.get(r["n_devices"], {}))
        fp = fp_by_n.get(r["n_devices"])
        if fp is not None:
            r["fingerprint"] = fp
    return {"geometry": geometry, "op": op, "size": size_label,
            "recon_shape": list(recon_shape), "rows": rows,
            "failures": failures, "skips": skips}


# ── Worker entry (internal; the orchestrator builds argv) ─────────────────────
def _probe_sharding_by_geom():
    """Per-GEOMETRY sharding capability of the library under test.

    True  = the geometry's projector supports the placement/sharded path (model._supports_sharding()),
            so it can be swept across multiple devices;
    False = single-device only (e.g. cone on prerelease, which raises on a multi-device request; or
            anything on pre-sharding main, which has no sharding API at all);
    None  = couldn't determine (probe failed) -> the orchestrator does NOT restrict.

    The orchestrator uses this to sweep each geometry only at the device counts it actually supports.
    """
    import mbirjax
    ang = np.linspace(0, np.pi, 16, endpoint=False)
    ang2 = np.column_stack([ang, np.zeros_like(ang)])
    tv16 = mbirjax.gen_translation_vectors(5, 3, 100.0, 100.0)   # 15 translations, tiny
    builders = {
        "parallel": lambda: mbirjax.ParallelBeamModel((16, 8, 8), ang),
        "cone": lambda: mbirjax.ConeBeamModel((16, 8, 8), ang,
                                              source_detector_dist=32.0, source_iso_dist=16.0),
        "multiaxis_parallel": lambda: mbirjax.MultiAxisParallelModel((16, 8, 8), ang2),
        "translation": lambda: mbirjax.TranslationModel((int(tv16.shape[0]), 8, 8), tv16,
                                                        source_detector_dist=32.0, source_iso_dist=16.0),
        "denoiser": lambda: mbirjax.QGGMRFDenoiser((8, 8, 8)),
    }
    out = {}
    for name, mk in builders.items():
        try:
            with contextlib.redirect_stdout(io.StringIO()):   # hush construction notices (e.g. TCT
                m = mk()                                       # geometry warnings on the tiny probe model)
            ss = getattr(m, "_supports_sharding", None)
            out[name] = bool(ss()) if callable(ss) else hasattr(m, "configure_devices")
        except Exception:   # noqa: BLE001 — a capability probe must never abort setup
            out[name] = None
    return out


def worker_setup(out_file):
    """Report platform + device count/label + per-geometry sharding capability."""
    import mbirjax  # noqa: F401  device-setup-first
    plat, max_dev = sc.detect_platform()
    dev_label = sc.device_label()
    corr = {"check": "no correctness fingerprint yet", "baseline_present": False}
    result = sc.build_setup_result(plat, max_dev, dev_label, corr)
    result["sharding_by_geom"] = _probe_sharding_by_geom()
    print(f"[setup] sharding_by_geom={result['sharding_by_geom']}")
    sc.write_worker_result(out_file, result)


def run_worker(argv):
    """Dispatch a --worker invocation (internal)."""
    p = argparse.ArgumentParser(description="performance_tracking worker (internal)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=["setup", "measure"], required=True)
    p.add_argument("--config", default=None, help="path to the serialized Config YAML")
    p.add_argument("--geometry", default=None)
    p.add_argument("--op", default=None)
    p.add_argument("--size", default=None, help="LxRxC")
    p.add_argument("--device-counts", type=int, nargs="+", default=None)
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    if a.mode == "setup":
        worker_setup(a.out_file)
        return
    config = Config.from_dict(sc.load_yaml(a.config))
    res = measure_cell_group(config, a.geometry, a.op, a.size, a.device_counts, a.out_file)
    sc.write_worker_result(a.out_file, res)   # final authoritative result (over run_measure_loop's)


# ── Orchestrator (default; touches no JAX unless inline) ──────────────────────
def _inline_setup(config):
    """Inline-mode platform probe.  The CPU device-count XLA flag is derived from
    MBIRJAX_NUM_CPU_DEVICES on the FIRST ``import mbirjax``, so set it BEFORE that import
    (setdefault respects a value the shell/cluster already set).  Ignored on GPU.
    """
    os.environ.setdefault("MBIRJAX_NUM_CPU_DEVICES", str(max(config.device_counts)))
    import mbirjax  # noqa: F401  device-setup-first
    plat, max_dev = sc.detect_platform()
    print(f"[inline setup] platform={plat}  max_devices={max_dev}  ({sc.device_label()})")
    return plat, max_dev


def _git_provenance(root):
    """{git_commit, git_branch, git_dirty} for the checkout at ``root`` (best-effort)."""
    import subprocess
    def _g(args):
        try:
            r = subprocess.run(["git", "-C", root, *args],
                               capture_output=True, text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    dirty = _g(["status", "--porcelain"]) or ""
    # Paths only (strip the two-char status + rename arrows); capped so a messy tree can't
    # bloat the header.  dirty_code distinguishes "the MEASURED code wasn't the commit"
    # from harmless doc/plan edits (a real case: a dirty .md-only tree muddied a memory-step
    # attribution on 2026-07-11).
    dirty_files = [ln[3:].split(" -> ")[-1] for ln in dirty.splitlines()]
    return {"git_commit": _g(["rev-parse", "HEAD"]),
            # committer date in strict ISO-8601, so the dashboard can place a run
            # on the timeline at the commit's time rather than the collection time
            # (lets older prerelease checkouts be added as past baselines).
            "git_commit_date": _g(["show", "-s", "--format=%cI", "HEAD"]),
            "git_branch": _g(["rev-parse", "--abbrev-ref", "HEAD"]),
            "mbirjax_version": sc.pyproject_version(root),
            "git_dirty": bool(dirty),
            "git_dirty_files": dirty_files[:20],
            "git_dirty_code": any(f.startswith("mbirjax/") for f in dirty_files)}


def _mbirjax_policy_defaults(root):
    """The library's EFFECTIVE reconstruction-policy defaults, captured from a real import in a
    subprocess (the orchestrator itself stays JAX-free).  Recorded in the run header so the
    dashboard can attribute a perf/memory step to a DEFAULT change (a real case: the
    partition_sequence default reverting [4,7] -> [0,2,4,6,7] via a merge read as a +20% memory
    step in every vcd_nonconst cell).  Best-effort: None if the import fails."""
    import subprocess, sys, json as _json
    snippet = (
        "import json, inspect, mbirjax\n"
        "from mbirjax._utils import _reconstruction_defaults_dict as d\n"
        "print(json.dumps({"
        "'partition_sequence': list(d['partition_sequence'].val), "
        "'granularity': list(d['granularity'].val), "
        "'max_iterations_default': inspect.signature(mbirjax.TomographyModel.recon).parameters['max_iterations'].default, "
        # The 2026-07 flash-remediation padding regime (per-end axial extension, split
        # h_recon formula + taper retirement, NSI cleanup — landed together on the branch):
        # get_support_radius is its capability marker, so this flag flips exactly when the
        # cone-cell shape/value/memory step appears.
        "'axial_extension': bool(getattr(mbirjax, 'get_support_radius', None))}))"
    )
    try:
        r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True,
                           timeout=180, cwd=root)
        return _json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else None
    except Exception:
        return None


def _file_tag(prov, fallback_date):
    """Filename tag = ``<commit-UTC-timestamp>_<sha8>``, so each run file is unique per commit and
    sorts chronologically by COMMIT time (not collection time).  Falls back to the collection date
    if commit info is absent (e.g. provenance lookup failed)."""
    import datetime as _dt
    sha = (prov.get("git_commit") or "")[:8]
    stamp = fallback_date
    cd = prov.get("git_commit_date")
    if cd:
        try:
            stamp = _dt.datetime.fromisoformat(cd).astimezone(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        except Exception:
            stamp = fallback_date
    return f"{stamp}_{sha}" if sha else stamp


# ── Record book (best-ever per cell/metric + the commit that set it) ───────────
# Categories tracked, and whether best is the MIN (time/memory) or MAX (speedup).
RECORD_METRICS = {"min_ms": "min", "mem_mb": "min", "speedup": "max"}


def update_records(records, cells, commit, date):
    """Update the cumulative best-per-(cell, metric) record book IN PLACE and annotate cells.

    ``records`` (loaded from records_<plat>.yaml, or {}) maps "geom|op|size|n_dev" -> per-metric
    {value, commit, date, prev}.  For each MEASURED cell, every RECORD_METRICS metric is compared
    against the stored best (min for time/memory, max for speedup): a first-ever value establishes
    a baseline (silent); a value that BEATS the prior best overwrites it (keeping prev) and is a
    "win" — the cell gains a ``new_records`` list naming the won metrics.  The trivial n=1 speedup
    (always 1.0) is excluded.  Returns ``(new_lines, n_baselines)`` for the run summary.
    """
    new_lines, n_baselines = [], 0
    for c in cells:
        if c.get("failed") or c.get("skipped"):
            continue
        key = f"{c['geometry']}|{c['op']}|{c['size']}|{c['n_devices']}"
        rec = records.setdefault(key, {})
        won = []
        for metric, direction in RECORD_METRICS.items():
            if metric not in c:
                continue
            if metric == "speedup" and c["n_devices"] == 1:
                continue   # trivially 1.0 at one device
            val = float(c[metric])
            cur = rec.get(metric)
            if cur is None:
                rec[metric] = {"value": val, "commit": commit, "date": date, "prev": None}
                n_baselines += 1
            elif (val < cur["value"] if direction == "min" else val > cur["value"]):
                new_lines.append(f"  NEW RECORD  {key}  {metric}={val:.4g} "
                                 f"(prev {cur['value']:.4g} @ {(cur.get('commit') or '?')[:8]})")
                rec[metric] = {"value": val, "commit": commit, "date": date,
                               "prev": cur["value"]}
                won.append(metric)
        if won:
            c["new_records"] = won
    return new_lines, n_baselines


# ── Diff + gate (compare a run vs its prior run; classify; set exit code) ──────
def _cell_key(c):
    return f"{c['geometry']}|{c['op']}|{c['size']}|{c['n_devices']}"


def _cell_status(c):
    """ok (measured) / failed / skipped / absent (None)."""
    if c is None:
        return "absent"
    if c.get("failed"):
        return "failed"
    if c.get("skipped"):
        return "skipped"
    return "ok"


def _expected_cells(result):
    """The (geom|op|size|n_dev) keys this run's config was supposed to attempt.

    Restricted per geometry by sharding_by_geom: a geometry that can't shard is only expected at
    n=1, so the gate doesn't flag its (legitimately unmeasured) multi-device cells as 'absent'.
    """
    cfg = result.get("config", {})
    plat = result["platform"]
    geom_sizes = cfg.get("geom_sizes", {}) or {}
    geom_ops = cfg.get("geom_ops", {}) or {}
    default_sizes = cfg.get("sizes", {}).get(plat, [])
    default_ops = cfg.get("ops", [])
    dc = result.get("device_counts", [])
    cap = result.get("sharding_by_geom", {})
    keys = set()
    for g in cfg.get("geometries", []):
        gs = (geom_sizes.get(g, {}) or {}).get(plat) or default_sizes
        sizes = [sc.size_label(s) for s in gs]   # per-geometry sizes (matches the orchestrator)
        ops = geom_ops.get(g) or default_ops     # per-geometry ops (matches the orchestrator)
        g_dc = [1] if cap.get(g) is False else dc
        for op in ops:
            for s in sizes:
                for n in g_dc:
                    keys.add(f"{g}|{op}|{s}|{n}")
    return keys


def _fmt_delta(today, ref, unit=""):
    """'<today> vs <expected> (<+abs>, <+pct>)' — shows BOTH the absolute and the % difference
    so a reader can judge importance (a big % on a tiny absolute is often noise, and vice versa)."""
    d = today - ref
    pct = (d / ref * 100.0) if ref else float("nan")
    return f"{today:g}{unit} vs {ref:g}{unit} expected ({d:+g}{unit}, {pct:+.1f}%)"


def _gate_fingerprint(key, tf, rf, op, lab, config, hard, soft):
    """Correctness gate on the tolerant fingerprint (see §7): exact shape/dtype, robust
    aggregates within rtol (HARD), a few sample deviations allowed (SOFT), new padding leak (HARD).
    Each aggregate finding shows the relative diff vs the tolerance plus the absolute change."""
    if not tf or not rf:
        return
    if tf.get("shape") != rf.get("shape"):
        hard.append(f"[{lab}] {key} fingerprint shape {rf.get('shape')} -> {tf.get('shape')}")
        return
    if tf.get("dtype") != rf.get("dtype"):
        hard.append(f"[{lab}] {key} fingerprint dtype {rf.get('dtype')} -> {tf.get('dtype')}")
    rtol = config.fp_rtol_iter if op in ("vcd_nonconst", "denoise") else config.fp_rtol_single
    for m in ("sum", "mean", "l2norm"):
        rv, tv = rf.get(m), tf.get(m)
        if rv is None or tv is None:
            continue
        reldiff = abs(tv - rv) / (abs(rv) or 1.0)
        if reldiff > rtol:
            hard.append(f"[{lab}] {key} fingerprint {m}: reldiff {reldiff:.2e} > rtol {rtol:g} "
                        f"(Δ {tv - rv:+.3g}; {tv:g} vs {rv:g} expected)")
    rs_, ts_ = rf.get("samples") or [], tf.get("samples") or []
    if rs_ and len(rs_) == len(ts_):
        dev = sum(1 for a, b in zip(rs_, ts_) if abs(b - a) / (abs(a) or 1.0) > rtol)
        if dev > config.k_sample_tol:
            soft.append(f"[{lab}] {key} {dev}/{len(rs_)} fingerprint samples deviate (rtol {rtol:g})")
    if tf.get("padding_zero") is False and rf.get("padding_zero") is not False:
        hard.append(f"[{lab}] {key} padding leak: padding_zero {rf.get('padding_zero')} -> False")


def _gate_metrics(key, t, r, lab, plat, config, hard, soft):
    """Metric gates for an ok->ok cell.  Structural changes and the correctness fingerprint are
    HARD on every platform.  Of the PERFORMANCE signals, only MEMORY is HARD, and only on GPU,
    where peak_bytes_in_use is ~deterministic (it is also what catches the gather-bug class —
    memory that fails to shard); on CPU memory is whole-process RSS (coarse) so it is SOFT.
    Speedup and absolute time are SOFT on every platform — both derive from timings, which are
    noisy even on GPU (especially small runs).  Every delta shows the value vs expected with BOTH
    the absolute and the percentage difference."""
    pre = f"[{lab}] {key} "
    # memory — HARD on GPU, SOFT (coarse RSS) on CPU.
    rm, tm = r.get("mem_mb"), t.get("mem_mb")
    if rm and tm is not None and (tm - rm) / rm * 100.0 > config.mem_hard_pct:
        bucket = hard if plat == "gpu" else soft
        cpu_note = "" if plat == "gpu" else " [CPU RSS, coarse]"
        bucket.append(pre + "memory " + _fmt_delta(tm, rm, " MB") + cpu_note)
    # speedup-ratio drop — SOFT everywhere (ratio of noisy timings).
    rsp, tsp = r.get("speedup"), t.get("speedup")
    if t["n_devices"] > 1 and rsp and tsp is not None and (rsp - tsp) / rsp * 100.0 > config.speedup_warn_pct:
        soft.append(pre + "speedup " + _fmt_delta(tsp, rsp))
    # absolute time — SOFT everywhere.
    rt, tt = r.get("min_ms"), t.get("min_ms")
    if rt and tt is not None and (tt - rt) / rt * 100.0 > config.time_soft_pct:
        soft.append(pre + "time " + _fmt_delta(tt, rt, " ms"))
    # structural — DIRECTION-AWARE.  Losing the sharded path (True->False) or the banded back path
    # (value->None), or a band-count change between two sharded runs, is a HARD regression.  GAINING
    # sharding (False->True) or banding (None->value) is the placement port LANDING — an improvement,
    # recorded SOFT, not gated.  (Without this, the run where sharding lands on a tracked branch flags
    # every newly-sharded cell as a regression — a huge false-positive burst.)
    ts, rs_ = bool(t.get("is_sharded")), bool(r.get("is_sharded"))
    if ts != rs_:
        msg = pre + f"is_sharded {r.get('is_sharded')} -> {t.get('is_sharded')}"
        (hard if (rs_ and not ts) else soft).append(msg + ("" if (rs_ and not ts) else " (gained sharding)"))
    tb, rb = t.get("back_n_bands_per_shard"), r.get("back_n_bands_per_shard")
    if tb != rb and (tb is not None or rb is not None):
        msg = pre + f"back band count {rb} -> {tb}"
        (soft if rb is None else hard).append(msg + (" (banded back-projection landed)" if rb is None else ""))
    _gate_fingerprint(key, t.get("fingerprint"), r.get("fingerprint"), t.get("op", ""),
                      lab, config, hard, soft)


def _compare_cell(key, t, r, lab, plat, expected, oom_gos, config, hard, soft):
    """Classify one cell vs one reference (see plan §10a status transitions)."""
    ts, rs = _cell_status(t), _cell_status(r)
    if rs == "absent":
        soft.append(f"[{lab}] new cell, no baseline (not gated): {key}")
        return
    if ts == "absent":
        gos = tuple(key.split("|")[:3])
        if key not in expected:
            soft.append(f"[{lab}] dropped from sweep: {key}")
        elif gos in oom_gos:
            soft.append(f"[{lab}] {key} not measured (OOM-descent stopped at higher n_dev)")
        else:
            hard.append(f"[{lab}] expected cell vanished (no row/skip/fail): {key}")
        return
    if ts == "failed" and rs == "ok":
        hard.append(f"[{lab}] {key} REGRESSED: was ok, now fails ({str(t.get('error',''))[:50]})")
        return
    if ts == "ok" and rs == "failed":
        soft.append(f"[{lab}] {key} improved: was failing, now ok")
        return
    if ts != "ok" or rs != "ok":   # skip<->fail combos, or unchanged fail/skip (quiet)
        if ts != rs:
            soft.append(f"[{lab}] {key} status {rs} -> {ts}")
        return
    _gate_metrics(key, t, r, lab, plat, config, hard, soft)   # ok -> ok


def gate_run(result, references, config):
    """Compare ``result`` against each (label, ref_result) and return the gate dict.

    Fires on a CHANGE vs the reference (plan §10/§10a): ok->fail / memory / speedup / structural /
    correctness are HARD; absolute time / added-dropped / improvements are SOFT; persistent
    failures are quiet.  Cold start (no usable reference) is all-SOFT, never a fail.
    """
    hard, soft = [], []
    refs = [(lab, r) for lab, r in references if r]
    if not refs:
        return {"result": "warn", "hard": [], "compared_to": [],
                "soft": ["no prior run to compare against (cold start) — nothing gated"]}
    plat = result.get("platform", "")
    expected = _expected_cells(result)
    oom_gos = {(c["geometry"], c["op"], c["size"])
               for c in result["cells"] if c.get("failed") and c.get("oom")}
    today = {_cell_key(c): c for c in result["cells"]}
    for lab, ref in refs:
        refcells = {_cell_key(c): c for c in ref.get("cells", [])}
        for key in sorted(set(today) | set(refcells)):
            _compare_cell(key, today.get(key), refcells.get(key), lab, plat,
                          expected, oom_gos, config, hard, soft)
    # ── DASHBOARD-MARKER CONTRACT (keep in sync) ──────────────────────────────────────────────────
    # Every HARD message above is "[lab] {key} ..." with key = _cell_key(c) = "geom|op|size|ndev".
    # The dashboard (build_dashboard.py `_parse_gate_hard` / `_GATE_CELL_RE`) extracts that cell id to
    # place the red marker on the scaling plot.  A hard message with NO parseable cell is still COUNTED
    # (gate tiles + history) but silently NOT MARKED — a count-vs-marker mismatch.  This should never
    # fire (all keys match the regex); if it does, the gate-string format drifted from the dashboard's
    # — reconcile BOTH sides.  ⚠ If you change _cell_key / the hard-string format, update _GATE_CELL_RE.
    _cell_pat = re.compile(r"[a-z_]+\|[a-z_]+\|\d+x\d+x\d+\|\d+")
    unmarkable = [h for h in hard if not _cell_pat.search(h)]
    if unmarkable:
        print(f"  [gate-marker WARNING] {len(unmarkable)} hard-gate message(s) carry no "
              f"dashboard-parseable cell id (geom|op|size|ndev) — they will be COUNTED but NOT "
              f"marked on the scaling plot.  Reconcile gate_run with build_dashboard.py _GATE_CELL_RE:")
        for h in unmarkable:
            print(f"      {h}")
    return {"result": "fail" if hard else ("warn" if soft else "pass"),
            "hard": hard, "soft": soft, "compared_to": [lab for lab, _ in refs]}


def _find_prior(out_dir, plat, current_tag):
    """Most-recent run file STRICTLY BEFORE current_tag (by name), or None.

    Filenames embed the commit-time tag, so a lexicographic sort is chronological by COMMIT time;
    the prior is therefore the immediately-preceding commit's run (not just 'yesterday's file').
    """
    import glob
    cur_name = f"regression_{plat}_{current_tag}.yaml"
    # Exclude the sibling *_table.yaml dumps: they match this glob and, since '_table.yaml' sorts AFTER
    # '.yaml', the prior run's table would otherwise be picked as the prior run (gate vs a non-run).
    befores = sorted(n for n in (os.path.basename(f)
                     for f in glob.glob(os.path.join(out_dir, f"regression_{plat}_*.yaml")))
                     if n < cur_name and not n.endswith("_table.yaml"))
    return os.path.join(out_dir, befores[-1]) if befores else None


def _print_gate(g):
    print("\n" + "=" * 78)
    print(f"  GATE: {g['result'].upper()}   (vs {', '.join(g['compared_to']) or 'nothing'})")
    print("=" * 78)
    for h in g.get("hard", []):
        print("  HARD  " + h)
    for s in g.get("soft", []):
        print("  warn  " + s)
    if not g.get("hard") and not g.get("soft"):
        print("  no changes vs reference")


def _print_summary(cells):
    """Per (geometry, op): min time (ms) / peak mem (MB) / speedup, for each (size, n_dev)."""
    print("\n" + "=" * 78)
    print("  REGRESSION SUMMARY — min time (ms) / peak mem (MB) / speedup vs fewest devices")
    print("=" * 78)
    groups = OrderedDict()
    for c in cells:
        groups.setdefault((c["geometry"], c["op"]), []).append(c)
    for (g, op), rows in groups.items():
        print(f"\n  {g} | {op}")
        print("  {:<12s}{:>6s}{:>11s}{:>11s}{:>9s}".format(
            "size", "n_dev", "min_ms", "mem_mb", "speedup"))
        for r in sorted(rows, key=lambda r: (r["size"], r["n_devices"])):
            if r.get("skipped"):
                print(f"  {r['size']:<12s}{r['n_devices']:>6d}   [skip] {r['reason']}")
                continue
            if r.get("failed"):
                tag = "OOM" if r.get("oom") else "FAIL"
                print(f"  {r['size']:<12s}{r['n_devices']:>6d}   [{tag}] {str(r.get('error', ''))[:58]}")
                continue
            mark = " !" if r.get("throttled") else ""
            print("  {:<12s}{:>6d}{:>11.1f}{:>11.1f}{:>8.2f}x{}".format(
                r["size"], r["n_devices"], r["min_ms"], r["mem_mb"],
                r.get("speedup", float("nan")), mark))


def run(config):
    """Run the full GEOMETRY × OP × size × device-count sweep and write the dated YAML."""
    if not config.out_dir:
        raise ValueError("Config.out_dir is required")
    if not config.date:
        raise ValueError("Config.date is required (stamp it in the orchestrator)")
    os.makedirs(config.out_dir, exist_ok=True)
    script = os.path.abspath(__file__)

    print("=" * 72)
    print(f"  performance_tracking — {'INLINE (single process)' if config.inline else 'isolated-subprocess'} harness")
    print(f"  beta root: {sc.beta_root()}")
    print("=" * 72)

    worker_env = None
    cfg_path = None
    if config.inline:
        plat, max_dev = _inline_setup(config)
        dev_label = sc.device_label()
        shard_by_geom = _probe_sharding_by_geom()
    else:
        worker_env = sc.build_worker_env(lib_root=config.lib_root or None)
        # Bound the CPU virtual-device count by THIS sweep (config.device_counts), not by
        # mbirjax's DEFAULT_MAX_CPU_DEVICES.  This MUST be set before the setup probe: the probe
        # imports mbirjax, and with no override mbirjax resolves only DEFAULT_MAX_CPU_DEVICES
        # devices -> detect_platform reports that as max_dev -> the sweep is silently capped (e.g.
        # 4 dropped when the library default is 2).  Harmless on GPU (CPU-backend flag only).
        worker_env["MBIRJAX_NUM_CPU_DEVICES"] = str(max(config.device_counts))
        setup, rc = sc.run_worker(script, ["--worker", "--mode", "setup"], extra_env=worker_env)
        if setup is None:
            print(f"  ERROR: setup worker produced no result (rc={rc}); aborting.")
            return None
        plat, max_dev, dev_label, _corr, _mpath = sc.print_setup_banner(setup)
        shard_by_geom = setup.get("sharding_by_geom", {})

    device_counts = [n for n in config.device_counts if n <= max_dev]
    # Per geometry: a projector that can't shard (cone on prerelease; anything on pre-sharding main;
    # translation/multiaxis until their placement ports land) runs single-device only — a multi-device
    # request RAISES.  So sweep each geometry only at the counts it supports (restrict to n=1 when
    # capability is explicitly False; never restrict on an unknown/None probe).
    def geom_device_counts(geom):
        return [1] if shard_by_geom.get(geom) is False else device_counts
    # Per-geometry SIZES + OPS: geom_sizes[geom][plat] / geom_ops[geom] override the shared
    # sizes[plat] / ops (the new translation/multiaxis geometries are kept small and skip vcd —
    # see Config.geom_sizes / geom_ops).
    def geom_size_labels(geom):
        gs = (config.geom_sizes.get(geom, {}) or {}).get(plat) or config.sizes[plat]
        return [sc.size_label(s) for s in gs]
    def geom_ops_for(geom):
        return config.geom_ops.get(geom) or config.ops
    print(f"  geometries: {config.geometries}")
    print(f"  device counts: {device_counts}   sharding_by_geom: {shard_by_geom}")
    for g in config.geometries:
        print(f"    {g}: ops={geom_ops_for(g)}  sizes={geom_size_labels(g)}")

    if not config.inline:
        fd, cfg_path = tempfile.mkstemp(suffix=".yaml", prefix="perf_cfg_")
        os.close(fd)
        sc.save_yaml(cfg_path, config.to_dict())

    cells = []
    swept_counts = set()
    for geometry in config.geometries:
        gdc = geom_device_counts(geometry)   # per-geometry device counts (n=1 only if it can't shard)
        swept_counts.update(gdc)
        size_labels = geom_size_labels(geometry)   # per-geometry sizes
        for op in geom_ops_for(geometry):          # per-geometry ops (new geometries skip vcd)
            for label in size_labels:
                print(f"\n=== {geometry} | {op} | {label} @ n={gdc} ===")
                if config.inline:
                    # Debug path: single step-through-able process; peak memory is CUMULATIVE
                    # across the device-count descent (see module docstring) -- trust the
                    # isolated-subprocess numbers, not these, for the memory ruler.
                    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="perf_inline_")
                    os.close(fd)
                    try:
                        res = measure_cell_group(config, geometry, op, label, gdc, tmp)
                    finally:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    per_count_results = [res] if res else []
                else:
                    # Isolate EACH device count in its own fresh worker so peak_bytes_in_use
                    # (a process-cumulative high-water mark with no in-process reset in jax) starts
                    # clean per config.  A shared-process descent lets one count's warmup/compile
                    # scratch inflate the next count's cumulative peak -- the source of the n>1
                    # memory-gate false positives.  Descend so the OOM early-stop still holds:
                    # fewer-device configs need MORE per-device memory, so once one OOMs the rest do.
                    per_count_results = []
                    for n in sorted(set(gdc), reverse=True):
                        args = ["--worker", "--mode", "measure", "--config", cfg_path,
                                "--geometry", geometry, "--op", op, "--size", label,
                                "--device-counts", str(n)]
                        res, _rc = sc.run_worker(script, args, extra_env=worker_env)
                        if not res:
                            # Hard worker death (OOM-killer / segfault / XLA abort) writes nothing,
                            # so run_worker returns None.  Record a VISIBLE failure marker (else this
                            # count silently vanishes from the cell) and STOP the descent: in the
                            # descending sweep fewer-device configs need MORE per-device memory, so a
                            # hard death here -- most often OOM -- would only repeat at smaller n.
                            print(f"  no result for {geometry}/{op}/{label} @ n={n} "
                                  f"(hard worker death, likely OOM); stopping descent")
                            per_count_results.append({"failures": [{
                                "n_devices": n, "oom": None,
                                "error": "worker produced no result (hard crash / likely OOM-killed)"}]})
                            break
                        per_count_results.append(res)
                        if any(f.get("oom") for f in (res.get("failures") or [])):
                            print(f"  stopping descent at {geometry}/{op}/{label}: n={n} OOMed; "
                                  f"fewer-device configs need more per-device memory")
                            break
                if not per_count_results:
                    print(f"  (no result for {geometry}/{op}/{label})")
                    continue
                # Merge the per-count workers back into one cell group.  Speedups are relative to
                # the fewest-device run WITHIN the group, so annotate only after all counts land.
                rows, group_failures, group_skips = [], [], []
                for res in per_count_results:
                    rows.extend(res.get("rows") or [])
                    group_failures.extend(res.get("failures") or [])
                    group_skips.extend(res.get("skips") or [])
                sc.annotate_speedups(rows)   # 'speedup' vs the fewest-device run in this group
                cells.extend(rows)
                for f in group_failures:
                    cells.append({"geometry": geometry, "op": op, "size": label,
                                  "n_devices": f["n_devices"], "failed": True,
                                  "oom": bool(f.get("oom")), "error": f.get("error")})
                for s in group_skips:
                    cells.append({"geometry": geometry, "op": op, "size": label,
                                  "n_devices": s["n_devices"], "skipped": True,
                                  "reason": s["reason"]})

    if cfg_path and os.path.exists(cfg_path):
        os.remove(cfg_path)

    prov = _git_provenance(config.lib_root or sc.beta_root())   # provenance of the LIBRARY under test
    # A worktree checkout of a specific ref (e.g. add_run) is DETACHED, so `rev-parse --abbrev-ref`
    # returns "HEAD".  Prefer the run_tag (which the nightly / add_run set to the real branch name)
    # in that case, so the run is labeled with its branch rather than "HEAD".
    if prov.get("git_branch") in (None, "", "HEAD") and config.run_tag:
        prov["git_branch"] = config.run_tag
    file_tag = _file_tag(prov, config.date)   # commit-time tag for the filename + prior selection

    # Update the cumulative record book (best per cell/metric + the commit that set it).  It lives
    # in out_dir, so nightly (results/regression/) and manual (results/manual/<tag>/) runs keep
    # SEPARATE books.  Done before writing the dated YAML so its cells carry the per-cell
    # `new_records` annotation; the records file itself, once versioned in mbirjax_metrics, has a
    # git history that IS the record-progression log.
    records_path = os.path.join(config.out_dir, f"records_{plat}.yaml")
    records = (sc.load_yaml(records_path) or {}) if os.path.exists(records_path) else {}
    new_lines, n_baselines = update_records(records, cells, prov.get("git_commit") or "?",
                                            config.date)
    sc.save_yaml(records_path, records)

    result = {
        "kind": "regression", "date": config.date, "platform": plat,
        "sharding_by_geom": shard_by_geom,
        "device_label": dev_label, **prov,
        "toolchain": sc.toolchain_info(),   # jax/jaxlib/CUDA stack — attributes a perf shift to the toolchain
        "packages": sc.installed_packages(),   # WHOLE env {name: version} — lets a deps-step name the dep that moved
        "dep_gen": config.dep_gen, "run_reason": config.run_reason,   # dependency-canary provenance
        "jax_available": config.jax_available or None,   # PyPI-latest jax at run time (None if unknown)
        # WHEN this run was measured (local ISO, like git_commit_date).  Stamped here in the orchestrator
        # (not a worker), once per run.  The dashboard plots a dep-canary re-run at this time rather than
        # the (older) commit time, so it doesn't stack on the commit's original point.
        "measured_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": config.to_dict(), "device_counts": sorted(swept_counts), "cells": cells,
        # Library policy defaults in effect for this run (see _mbirjax_policy_defaults).
        "policy": _mbirjax_policy_defaults(config.lib_root or sc.beta_root()),
    }

    # Diff + gate: compare against this branch's most-recent prior run (the immediately-preceding
    # commit's dated file in out_dir), classify per the §10/§10a rules, and stash the gate dict in
    # the YAML.  Done before the write so the dated file records its own verdict; the exit code is
    # set by main() from result.gate.  Prior-run is the SOLE gate reference — cross-branch
    # comparison (vs main/prerelease) and best-ever drift are surfaced on the dashboard, not gated.
    # dep_gen>0 (a dependency-canary re-run of this commit with a different dep set) appends `_gNNNN` so
    # it doesn't overwrite the commit's other-deps run; gen 0 keeps the historical name.  `_gNNNN` sorts
    # AFTER `.yaml`, so _find_prior still returns the correct immediately-preceding run (plan §3).
    gen_tag = f"{file_tag}_g{config.dep_gen:04d}" if config.dep_gen else file_tag
    gate_dict = None
    if config.compare_to_prior:
        refs = []
        pp = _find_prior(config.out_dir, plat, gen_tag)
        if pp:
            refs.append((f"prior:{os.path.basename(pp)}", sc.load_yaml(pp)))
        gate_dict = gate_run(result, refs, config)
        result["gate"] = gate_dict

    out_path = os.path.join(config.out_dir, f"regression_{plat}_{gen_tag}.yaml")
    sc.save_yaml(out_path, result)
    # Browsable companion next to the run YAML: <name>_table.yaml, a geometry/op/size/n table of this
    # run.  Written here so it lands in results/ and the nightly's `git add results` commits+pushes it
    # alongside the data.  Best-effort: a formatting hiccup must never fail the measurement run.
    try:
        import regression_to_table
        regression_to_table.write_table(regression_to_table.load_yaml(out_path),
                                        os.path.splitext(out_path)[0] + "_table.yaml")
    except Exception as e:   # noqa: BLE001
        print(f"[warn] companion _table.yaml not written: {e}")
    _print_summary(cells)
    if new_lines:
        print(f"\n  {len(new_lines)} NEW RECORD(S) this run:")
        for line in new_lines:
            print(line)
    elif n_baselines:
        print(f"\n  established {n_baselines} baseline record(s) (first run for these cells)")
    if gate_dict:
        _print_gate(gate_dict)
    print(f"\nOutput written to: {out_path}")
    print(f"Record book:       {records_path}")
    print("Done.")
    return result


def main():
    """Default entry: the nightly config, dated today, into results/regression/.

    Exit code: non-zero on a HARD-gate regression when gating is on, so a cron/slurm wrapper
    surfaces it as a real alert.
    """
    from datetime import datetime
    config = Config()
    config.out_dir = os.path.join(sc.RESULTS_DIR, "regression")
    config.date = datetime.now().strftime("%Y%m%d")
    result = run(config)
    if config.gate and result and (result.get("gate") or {}).get("result") == "fail":
        sys.exit(1)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
