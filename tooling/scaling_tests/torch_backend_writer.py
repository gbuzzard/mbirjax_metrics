"""
tooling/scaling_tests/torch_backend_writer.py
─────────────────────────────────────────────
Nightly / manual REGRESSION engine for the PyTorch backend (mbirtorch).  Writes
``results/{gpu,cpu}-torch/<branch>/regression_<plat>_<commit-tag>.yaml`` plus the
sibling ``_table.yaml``, ``records_<plat>.yaml`` and ``tests_<plat>_<date>.txt`` —
the torch rows the dashboard's backend design consumes (its own History row, the
backend-qualified Platform entries).

This is a torch MEASUREMENT layer over ``performance_tracking``'s DECISION layer.
It imports that module and reuses its Config, its correctness fingerprint, its
gate, its record book, its prior-run selection, its rolling-min memory window and
its commit-time file tag.  Only the parts that must touch torch live here: model
construction, device pinning, the op bodies, the timing loop and the memory read.

Why not a private gate: two gate implementations drift, and the drift is
invisible until a torch regression is silently not caught.  One gate model means
the memory threshold, the fingerprint tolerances, the status-transition rules and
the cold-start rule are defined once.  The split is possible because
``scaling_common`` defers every jax import into the functions that need it, so
``import performance_tracking`` from a torch env initialises no jax backend.
The torch env must supply ruamel.yaml and matplotlib (scaling_common imports them
at module level).

Design record: ``mbirjax_plans/plans/torch_port/nightly_plan.md``.

THE OPS ARE THE JAX ENGINE'S OPS.  ``forward`` is ``sparse_forward_project`` over
the full ROR pixel set, ``back`` is ``sparse_back_project``, and ``vcd_nonconst``
is ``_vcd_recon`` (the jax engine's ``vcd_recon``) with the partitions built
OUTSIDE the timed region — not the user-facing whole-volume calls, and not
``recon()`` with its initialisation timed.
The inputs are the engine's own generators at the engine's own seeds.  An earlier
version of this file measured different operations under the same names, which
made the adjacent torch and jax dashboard rows an apples-to-oranges comparison.

Roles (mirrors performance_tracking.py):
  - orchestrator (default, no args)   : run() — per (geom, op, size) spawn a worker,
                                        collect rows, gate, write YAML.
  - worker --mode measure ...         : measure one cell group (all device counts).

Env vars (set by tooling/regression/run_torch_regression.sh):
  REG_TORCH_LIB_ROOT   (required)  the mbirtorch checkout under test (PYTHONPATH + provenance)
  REG_TORCH_OUT_DIR    (required)  results/<plat>/<branch_slug>/
  REG_TORCH_PLATFORM   (required)  'gpu-torch' | 'cpu-torch' — DECLARED, then verified
  REG_TORCH_DATE       (optional)  YYYYMMDD, resolved once by the wrapper
  REG_TORCH_GATE       (optional)  '1' (default) to exit non-zero on a hard regression
  REG_TORCH_RUN_TAG    (optional)  branch name recorded in the YAML
  REG_TORCH_DEVICE_COUNTS (optional)  space-separated, default '1'; n>1 applies only at the
                                      MULTI_DEVICE_SIZE_LABELS cells — every other cell stays n=1
  REG_TORCH_MEM_GATE_WINDOW (optional) rolling-min window in runs; default 1 (see build_config)
  REG_TORCH_SMOKE      (optional)  '1' -> a toy 1-cell sweep, for plumbing checks
"""
import argparse
import contextlib
import datetime
import gc
import io
import os
import platform as _platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import scaling_common as sc                # noqa: E402  jax-free at module level
import performance_tracking as pt          # noqa: E402  ditto (the decision layer)


# ── Cone recon_shape pins — padding-policy decoupling ─────────────────────────
# Same rationale as performance_tracking.CONE_RECON_SHAPE_PINS, but mbirtorch's OWN
# shapes: pin them so a future change to mbirtorch's axial-padding policy cannot
# silently move a cone cell's memory and time baselines.  The values below are
# mbirtorch's current auto-derived shapes, so pinning moves no baseline.
#
# NOTE these differ from the jax pins: mbirtorch's cone recon equals its parallel
# recon, while mbirjax's carries the flash-remediation axial extension (e.g.
# (200,208,160) -> (160,160,208) here vs (160,160,234) there).  So the cone cells
# are genuinely different problem sizes on the two backends, over and above any
# implementation difference.  Read the cone rows within a backend, not across.
CONE_RECON_SHAPE_PINS = {
    (128, 112, 96):    (96, 96, 112),
    (129, 113, 97):    (97, 97, 113),
    (200, 208, 160):   (160, 160, 208),
    (512, 448, 384):   (384, 384, 448),
    (513, 449, 385):   (385, 385, 449),
    (1024, 1008, 992): (992, 992, 1008),
}

# Sinogram sizes per platform key — the harness's own cells (port_plan.md §4), so the
# torch and jax rows sit at identical coordinates.  Keyed by the FULL platform key,
# because performance_tracking._expected_cells looks them up by result['platform'].
SIZES = {
    "gpu-torch": [(200, 208, 160), (512, 448, 384), (513, 449, 385), (1024, 1008, 992)],
    "cpu-torch": [(128, 112, 96), (129, 113, 97), (200, 208, 160)],
}
DENOISER_SIZES = {
    "gpu-torch": [(225, 241, 257), (512, 448, 384), (1024, 1008, 992)],
    "cpu-torch": [(128, 144, 160), (225, 241, 257)],
}

# Sizes that sweep MULTIPLE device counts (when REG_TORCH_DEVICE_COUNTS asks for them).
# The multi-GPU rows exist at the two sizes the torch campaign gates on, where multi-device
# history is directly comparable to the campaign record; the smaller sizes measure mostly
# communication overhead at n>1 and stay single-device (nightly_plan.md §3(c)).  The denoiser
# stays single-device at every size: QGGMRFDenoiser.denoise raises under any non-trivial
# placement, and the device-policy work deliberately leaves it outside the widening.
MULTI_DEVICE_SIZE_LABELS = {"512x448x384", "1024x1008x992"}


def cell_device_counts(geometry, size_label, device_counts):
    """The device counts one (geometry, op, size) cell group sweeps."""
    if geometry == "denoiser":
        return [1]
    if size_label in MULTI_DEVICE_SIZE_LABELS:
        return list(device_counts)
    return [1]


# The automatic-device-choice check, one per multi-GPU night.  Every measured
# row pins its device count, and a pin bypasses the automatic choice entirely,
# so the path a multi-GPU user hits by default -- the library choosing how many
# devices to use -- would otherwise never run on real hardware on a schedule.
# The check builds one UNPINNED model, lets the settle choose, and compares the
# realized count against what the shipped widening floors say it should be.
# The cell is the 512-class cone reconstruction, because its expected choice
# there is a MIDDLE count (the floors admit two devices and hold four on a
# four-GPU node), so the floors, their ordering, and the capacity search all
# participate in one check.  The expected count is computed at run time from
# the shipped floors table, never hardcoded, so a floors refresh moves the
# expectation with it and the check fails only when the realized choice
# disagrees with the table that shipped.  The verdict is recorded under its
# own key in the run file rather than as a measured row, so it cannot collide
# with the pinned rows' (geometry, op, size, n_devices) coordinates.
AUTO_CHOICE_GEOMETRY = "cone"
AUTO_CHOICE_SIZE = (512, 448, 384)


def build_config(platform_key, out_dir, date, run_tag, lib_root, device_counts, gate):
    """The nightly torch sweep as a performance_tracking.Config.

    Building a REAL Config (not a hand-rolled dict) is what makes the run file
    carry the five gate-threshold keys the dashboard reads for its threshold
    explanation, and what lets _expected_cells reconstruct the sweep this run was
    supposed to attempt.
    """
    cfg = pt.Config(
        geometries=["parallel", "cone", "denoiser"],
        ops=["direct_filter", "forward", "back", "vcd_nonconst"],
        geom_ops={"denoiser": ["denoise"]},
        sizes={platform_key: [list(s) for s in SIZES[platform_key]]},
        geom_sizes={"denoiser": {platform_key: [list(s) for s in DENOISER_SIZES[platform_key]]}},
        device_counts=list(device_counts),
        out_dir=out_dir, date=date, run_tag=run_tag, lib_root=lib_root, gate=gate,
    )
    # Rolling-min memory window.  The jax default of 3 exists for a jax artefact: its
    # sharded-path peak_bytes_in_use is bimodal per run (n=2 wandered ~12%, n=1 and n=4
    # were byte-frozen).  torch reads max_memory_allocated, a different instrument, and
    # this sweep is n=1 — so the default here is 1 (single-shot, no detection lag) until
    # the trial run's repeat ablation says otherwise.  nightly_plan.md §3(c-ii).
    cfg.mem_gate_window = int(os.environ.get("REG_TORCH_MEM_GATE_WINDOW") or 1)
    return cfg


# ── Guards ────────────────────────────────────────────────────────────────────
def assert_platform(declared):
    """Abort unless the hardware matches the platform key the WRAPPER declared.

    The platform key is declared, never inferred.  Inferring it from
    torch.cuda.is_available() cannot fail loudly: a GPU night on which CUDA did not
    initialise would quietly file itself under cpu-torch, and the gpu-torch charts
    would go silent with no other symptom.  That is exactly the 2026-07-21 failure
    in which the jax nightly measured the whole GPU suite on CPU and filed it under
    results/gpu/ (see performance_tracking._assert_platform_matches_out_dir, whose
    'gpu'/'cpu' membership test does not cover the hyphenated torch keys).
    """
    import torch
    if declared not in ("gpu-torch", "cpu-torch"):
        raise SystemExit(f"REG_TORCH_PLATFORM must be gpu-torch or cpu-torch, got {declared!r}")
    on_gpu = bool(torch.cuda.is_available())
    want_gpu = (declared == "gpu-torch")
    if on_gpu != want_gpu:
        raise RuntimeError(
            "PLATFORM MISMATCH: the wrapper declared REG_TORCH_PLATFORM={d!r} but "
            "torch.cuda.is_available() is {a}.\n"
            "  Measuring on one platform and filing under the other would write "
            "records_{d}.yaml into a tree the dashboard reads for the OTHER platform, "
            "and the charts would simply go quiet.  Aborting instead.\n"
            "  If this is a GPU night: check that the node actually allocated a GPU "
            "(no --gpus-per-node, or a CUDA/driver mismatch in the torch build).".format(
                d=declared, a=on_gpu))
    if want_gpu and torch.cuda.device_count() < 1:
        raise RuntimeError("PLATFORM MISMATCH: cuda is available but device_count() is 0.")


def assert_no_calibration():
    """Refuse to measure with mbirtorch's memory-calibration mode on.

    That mode calls reset_peak_memory_stats at the top of _vcd_recon and OWNS the peak
    counter, so it would clobber the very number these rows record.  Checked rather
    than trusted, because it is an ambient environment variable.
    """
    val = os.environ.get("MBIRTORCH_MEMORY_CALIBRATION")
    if val and val.lower() not in ("", "0", "false"):
        raise RuntimeError(
            f"MBIRTORCH_MEMORY_CALIBRATION={val!r} is set.  That mode resets and owns "
            "torch.cuda.max_memory_allocated, which is the memory ruler these rows read. "
            "Unset it before measuring.")


# ── Devices: pin explicitly, then verify what was BOUND ───────────────────────
def pin_devices(model, n, platform_key):
    """Pin the model to EXACTLY n devices of the platform's kind and return the
    realized device list.

    The pin is mandatory on every row, at every count including n=1.  mbirtorch's
    device policy (landed 2026-08-08) gives an unpinned model an all-device default
    on multi-GPU CUDA, with the single device resolved lazily as cuda > mps > cpu.
    Without this call every row would silently measure an all-device run under a
    cell labelled n=1 — the same defect the device-policy design records in
    p4_gate_readout.py, where the n=1 arm was also the reference the value diffs
    were taken against.

    The pin must also name the device KIND, not just the count.  On a Mac,
    configure_devices(num_devices=1) binds the lazily-preferred device, which is
    MPS — so a cpu-torch row pinned by count alone would silently measure Apple's
    GPU and file it under cpu.  cpu-torch therefore pins devices=['cpu'] explicitly
    (repeated virtual cpu devices at n>1, as mbirtorch's own sharding tests do).

    configure_devices() sets device_layout_is_automatic=False permanently, which is
    the flag the automatic path consults, so it is the durable pin.  It also rebuilds
    the placements and recreates the projectors, so it must run BEFORE any warmup.

    Verification is the layer that makes the pin checkable: read back what the model
    actually bound — count AND kind — and refuse to measure on any disagreement (the
    arm-check discipline of phase5_findings.md, applied to device binding).
    """
    want_kind = "cuda" if platform_key == "gpu-torch" else "cpu"
    if want_kind == "cuda":
        # n=1 binds cuda:0: assert_platform already proved CUDA is up on this key.
        model.configure_devices(num_devices=n)
    else:
        model.configure_devices(devices=["cpu"] * n)
    devices = list(model.sino_placement.devices)
    if len(devices) != n:
        raise RuntimeError(
            f"DEVICE PIN FAILED: asked for {n} device(s), model bound {len(devices)} "
            f"({[str(d) for d in devices]}).  Refusing to file this row under n={n}.")
    wrong = [str(d) for d in devices if d.type != want_kind]
    if wrong:
        raise RuntimeError(
            f"DEVICE PIN FAILED: platform {platform_key} wants only {want_kind} "
            f"devices, model bound {wrong}.  Refusing to file this row under "
            f"{platform_key}.")
    return devices


def placement_info(model, devices):
    """The per-row record of WHAT WAS BOUND (not what was requested)."""
    return {"is_sharded": not bool(model.sino_placement.is_trivial),
            "n_shard_devices": int(model.sino_placement.n_devices),
            "devices": [str(d) for d in devices]}


# ── Model + inputs ────────────────────────────────────────────────────────────
def make_model(config, geometry, size, platform_key):
    """Build an mbirtorch model of ``geometry`` for SINOGRAM ``size``.

    Mirrors performance_tracking.make_model: the same cone geometry convention
    (magnification 2 via source_detector_dist = 4 * channels), the same recon-shape
    pinning for cone, verbose off.  The constructors take no device argument
    (configure_devices is the single door since the 2026-08-08 policy landing), so
    every model built here MUST be followed by pin_devices before any use — an
    unpinned model resolves its device lazily and, on multi-GPU CUDA, auto-widens.
    """
    import mbirtorch
    n_views, n_rows, n_channels = size
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    if geometry == "parallel":
        model = mbirtorch.ParallelBeamModel((n_views, n_rows, n_channels), angles)
    elif geometry == "cone":
        sdd = config.cone_sdd_over_channels * n_channels
        model = mbirtorch.ConeBeamModel((n_views, n_rows, n_channels), angles,
                                        source_detector_dist=sdd, source_iso_dist=sdd / 2.0)
        pin = CONE_RECON_SHAPE_PINS.get((int(n_views), int(n_rows), int(n_channels)))
        if pin is not None:
            model.set_params(recon_shape=tuple(int(x) for x in pin), no_warning=True)
        else:
            print(f"WARNING: cone size {(n_views, n_rows, n_channels)} has no recon_shape pin; "
                  f"using auto {tuple(int(x) for x in model.get_params('recon_shape'))} — add it "
                  f"to CONE_RECON_SHAPE_PINS to decouple from padding policy.", file=sys.stderr)
    elif geometry == "denoiser":
        model = mbirtorch.QGGMRFDenoiser(tuple(int(x) for x in size))
        model.set_params(sharpness=config.denoise_sharpness, no_warning=True)
    else:
        raise ValueError(f"unknown geometry {geometry!r} (expected parallel/cone/denoiser)")
    model.set_params(verbose=0, no_warning=True)
    return model


def make_indices(model):
    """Full field-of-view pixel indices — the jax engine's definition, verbatim."""
    import mbirtorch
    recon_shape = model.get_params('recon_shape')
    return mbirtorch.gen_full_indices(tuple(int(x) for x in recon_shape),
                                      use_ror_mask=model.get_params('use_ror_mask'))


def to_device(model, arr, kind):
    """Pre-place a HOST input in the model's device form, OUTSIDE the timing loop.

    Measure the op, not the host->device transfer.  ``kind`` is 'sino' (view axis)
    or 'recon' (slice axis); at n=1 both are a plain tensor on the model's device.
    """
    import torch
    if kind == "sino":
        placed = model._shard_sinogram(arr)
    else:
        placed = model._shard_recon(arr)
    _sync(model)
    return placed


def to_numpy(out):
    """Device form (tensor or Shards) -> numpy, for the fingerprint.

    Test for a tensor FIRST: torch.Tensor also has a .gather method (the indexing
    one), so duck-typing on 'gather' alone calls that with no arguments.

    Shards.gather() already returns NUMPY (it detaches and concatenates on the host
    internally), so it must not be detached again — doing so raised
    "'numpy.ndarray' object has no attribute 'detach'" on every n>1 row of the first
    multi-device trial.  The n=1 path never reaches this branch, which is why the
    single-device verification could not have caught it.
    """
    import torch
    if torch.is_tensor(out):
        return out.detach().cpu().numpy()
    if hasattr(out, "tensors") and hasattr(out, "placement"):   # _sharding.Shards
        return np.asarray(out.gather())
    return np.asarray(out)


def _sync(model):
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ── Op bodies — the jax engine's definitions ──────────────────────────────────
def run_filter(model, sino):
    return model.direct_filter(sino, output_sharded=True)


def run_forward(model, cylinders, pixel_indices):
    return model.sparse_forward_project(cylinders, pixel_indices)


def run_back(model, sino, pixel_indices):
    return model.sparse_back_project(sino, pixel_indices)


def build_partitions(model, sino_np, weights, max_iterations, seed):
    """Build the VCD partitions + sequence once, OUTSIDE the timing loop.

    gen_pixel_partition draws from the un-seeded global RNG, so without the seed the
    partitions — and therefore the recon — vary run to run and the day-over-day VCD
    fingerprint would false-positive.  The jax engine seeds for the same reason.
    """
    np.random.seed(seed)
    ret = model.initialize_recon(sino_np, weights=weights, max_iterations=max_iterations)
    return ret[3], ret[4]        # partitions, partition_sequence


def run_vcd(model, sino_np, weights, partitions, partition_sequence, measure_seed):
    """Timed op: one full VCD reconstruction with NONCONSTANT weights."""
    np.random.seed(measure_seed)
    recon, _stats = model._vcd_recon(sino_np, partitions, partition_sequence,
                                     stop_threshold_change_pct=0.0,
                                     weights=weights, init_recon=None)
    return recon


def run_denoise(model, image, config):
    np.random.seed(config.measure_seed)
    out, _ = model.denoise(image, sigma_noise=config.denoise_sigma,
                           max_iterations=config.denoise_iterations,
                           stop_threshold_change_pct=0.0, output_sharded=True)
    return out


# ── Timing + memory ───────────────────────────────────────────────────────────
def time_op(model, run_fn, warmup, trials, devices=None):
    """Time run_fn() over warmup + trials iterations, synchronising each result,
    and read the peak memory PER ITERATION.

    Mirrors scaling_common.time_op, including the memory discipline: drop the PREVIOUS
    iteration's result before allocating the next, so the device peak reflects a single
    call (input + one output) rather than two outputs alive at once.  gc.collect() sits
    outside the timed region so it cannot perturb the timing.

    The peak counters are reset before EVERY iteration and read after it, so each
    reading covers exactly one call.  One reading spanning the whole loop cannot say
    which call carried the peak: a 2026-08-19 comparison found a four-device arm
    recording a 26.6 GiB lead-device watermark where a fresh single reconstruction
    peaks at 6.84 GiB, and the spanning read could not localize it (the mechanism
    remains open; open item G4 in the plans repository).  Per-iteration readings make
    the column mean "one call's peak" and make an inflated iteration visible by
    itself.  On CPU the reset is a no-op and each reading is whole-process RSS, so
    cpu rows keep their coarse cumulative semantics; mem_kind says which ruler
    applied.

    Returns (stats, result, mem): mem carries the warmup iterations' peaks and the
    trial iterations' peaks in MB, in order, plus the ruler's name.
    """
    result = None
    times = []
    mem = {"peaks_warmup_mb": [], "peaks_trial_mb": [], "mem_kind": "n/a"}
    for i in range(warmup + trials):
        result = None
        gc.collect()
        if devices is not None:
            reset_peak_memory(devices)
        t0 = time.perf_counter()
        result = run_fn()
        _sync(model)
        dt = time.perf_counter() - t0
        if devices is not None:
            peak_mb, mem["mem_kind"] = peak_memory_mb(devices)
            key = "peaks_trial_mb" if i >= warmup else "peaks_warmup_mb"
            mem[key].append(round(float(peak_mb), 1))
        if i >= warmup:
            times.append(dt)
    arr = np.array(times) * 1e3
    return ({"min_ms": float(arr.min()), "mean_ms": float(arr.mean()),
             "std_ms": float(arr.std())}, result, mem)


def reset_peak_memory(devices):
    import torch
    if torch.cuda.is_available():
        for d in devices:
            if d.type == "cuda":
                torch.cuda.reset_peak_memory_stats(d)


def peak_memory_mb(devices):
    """Peak memory in MB over the row's PINNED devices, and the ruler's name.

    GPU: max over the pinned devices of torch.cuda.max_memory_allocated — ALLOCATED,
    not reserved, which is the ruler port_plan.md §3 names for the torch series and the
    counterpart of jax's peak_bytes_in_use.  Reading only the pinned devices (rather
    than every visible one, as mbirtorch.get_memory_stats does) keeps an n=2 row from
    picking up a neighbour's allocation.
    CPU: whole-process RSS, coarse — which is why the memory gate is soft there.
    """
    import torch
    if torch.cuda.is_available():
        peak = 0
        for d in devices:
            if d.type == "cuda":
                peak = max(peak, int(torch.cuda.max_memory_allocated(d)))
        return peak / (1024 ** 2), "gpu_peak_per_device"
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 ** 2) if _platform.system() == "Darwin" else rss / 1024
    return rss_mb, "cpu_rss"


def run_measure_loop(size_label, device_counts, out_file, build_and_time, header_extra=""):
    """Device-count descent for one problem size — the torch counterpart of
    scaling_common.run_measure_loop (which cannot be reused: it picks devices through
    jax).  Same semantics: descend so per-device allocation ascends within this fresh
    process, stop the descent on an OOM (fewer devices need MORE per-device memory),
    publish incrementally so a hard crash still returns the completed configs, sample
    GPU clocks/temps during each timed region, and free between configs.
    """
    desc = sorted(set(device_counts), reverse=True)
    print(f"\n[measure {size_label}{header_extra}]  device counts (descending): {desc}")
    rows, failures = [], []
    mem_kind = "n/a"

    def _publish():
        sc.write_worker_result(out_file, {"size": size_label, "mem_kind": mem_kind,
                                          "rows": rows, "failures": failures})

    gpu_present = bool(sc.sample_gpu_health())
    for n in desc:
        sampler = sc._GpuSampler().start() if gpu_present else None
        try:
            timed = build_and_time(n)
        except Exception as e:                    # noqa: BLE001 — never abort the sweep
            if sampler:
                sampler.stop()
            import traceback
            tb = traceback.format_exc()
            oom = sc.is_oom(tb)
            failures.append({"n_devices": n, "oom": oom,
                             "error": str(e).replace("\n", " ")[:300], "traceback": tb})
            print(f"  n_devices={n:2d}  {'OOM' if oom else 'ERROR'}: {str(e)[:120]}")
            if not oom:
                print(tb)
            _publish()
            if oom:
                print(f"  stopping descent at {size_label}: fewer-device configs need "
                      f"more per-device memory and would also OOM")
                break
            continue
        if sampler:
            sampler.stop()
        stats, mem_mb, mem_kind, extra = timed
        gpu_health = (sampler.worst() if sampler else []) or sc.sample_gpu_health()
        hot = sc.throttled_gpus(gpu_health)
        rows.append({"n_devices": n, **stats, "mem_mb": mem_mb,
                     "gpu_health": gpu_health, "throttled": bool(hot), **extra})
        print(f"  n_devices={n:2d}  min={stats['min_ms']:9.1f} ms  "
              f"mean={stats['mean_ms']:9.1f} ms  mem={mem_mb:8.1f} MB ({mem_kind})")
        if hot:
            print("  !! THROTTLING — this timing is UNRELIABLE: "
                  + ", ".join(sc._fmt_hot_gpu(g) for g in hot))
        _publish()
        gc.collect()
    _publish()
    return rows, failures


# ── Worker body ───────────────────────────────────────────────────────────────
def measure_cell_group(config, geometry, op, size_label, device_counts, platform_key, out_file):
    """Measure one (geometry, op, size) across ``device_counts``."""
    import mbirtorch  # noqa: F401
    assert_no_calibration()
    size = pt.parse_size_label(size_label)
    is_denoiser = (geometry == "denoiser")

    # Inputs come from the JAX ENGINE's generators at its seeds — pure numpy, so they
    # are reusable verbatim and the two backends see the same arrays.
    sino_np = None if is_denoiser else pt.make_sinogram(config, size)
    image_np = pt.make_noisy_image(config, size) if is_denoiser else None

    base_model = make_model(config, geometry, size, platform_key)
    pin_devices(base_model, 1, platform_key)
    recon_shape = tuple(int(x) for x in base_model.get_params('recon_shape'))
    if is_denoiser:
        idx = cylinders = num_pixels = None
    else:
        idx = make_indices(base_model)
        num_pixels = len(idx)
        cylinders = (pt.make_cylinders(num_pixels, recon_shape[2], config.input_seed)
                     if op == "forward" else None)
    weights = pt.make_weights(config, size) if op == "vcd_nonconst" else None
    del base_model
    gc.collect()

    # TRUE (unpadded) output shape per op, for the fingerprint crop.
    op_true_shape = {
        "direct_filter": tuple(size),
        "forward": tuple(size),
        "back": (num_pixels, recon_shape[2]),
        "vcd_nonconst": tuple(recon_shape),
        "denoise": tuple(recon_shape),
    }.get(op, tuple(size))

    trials = 1 if size_label in config.single_trial_sizes else config.trials_by_op.get(op, 3)

    def build_and_time(n):
        model = make_model(config, geometry, size, platform_key)
        devices = pin_devices(model, n, platform_key)   # pin FIRST, before any warmup
        info = placement_info(model, devices)
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
            partitions, partition_sequence = build_partitions(
                model, sino_np, weights, config.vcd_iterations, config.measure_seed)
            run_fn = lambda: run_vcd(model, sino_np, weights, partitions,
                                     partition_sequence, config.measure_seed)
        elif op == "denoise":
            run_fn = lambda: run_denoise(model, image_np, config)
        else:
            raise ValueError(f"op {op!r} not implemented")
        # The memory ruler lives INSIDE the timing loop: the counters reset
        # before every iteration and are read after it, so the row's number is
        # the largest single-call peak among the warm trials, and the warmup's
        # own peaks (which include the compiles) are recorded beside it rather
        # than folded in.
        stats, result, mem = time_op(model, run_fn, config.warmup, trials,
                                     devices=devices)
        mem_kind = mem["mem_kind"]
        trial_peaks = mem["peaks_trial_mb"]
        mem_mb = max(trial_peaks) if trial_peaks else 0.0
        # Re-verify the binding AFTER the timed call: a widening that happened inside
        # recon() would not be visible at pin time.
        if int(model.sino_placement.n_devices) != n:
            raise RuntimeError(
                f"DEVICE COUNT CHANGED DURING THE OP: pinned {n}, now "
                f"{int(model.sino_placement.n_devices)}.  Refusing to file under n={n}.")
        # Fingerprint AFTER the memory read, so the host gather cannot inflate the peak.
        fp = pt.fingerprint(to_numpy(result), op_true_shape)
        return stats, mem_mb, mem_kind, {**info, "fingerprint": fp,
                                         "platform": platform_key,
                                         "mem_peaks_warmup_mb":
                                             mem["peaks_warmup_mb"],
                                         "mem_peaks_trial_mb": trial_peaks}

    rows, failures = run_measure_loop(
        size_label, device_counts, out_file, build_and_time,
        header_extra=f" | {geometry} | op={op} | recon={recon_shape}")
    for r in rows:
        r["geometry"] = geometry
        r["op"] = op
        r["size"] = size_label
        r["recon_shape"] = list(recon_shape)
        r["trials"] = trials
    return {"geometry": geometry, "op": op, "size": size_label,
            "recon_shape": list(recon_shape), "rows": rows, "failures": failures}


def auto_choice_check(config, platform_key):
    """One UNPINNED settle on real devices, judged against the shipped floors.

    Runs in its own worker process like every other job here.  The model is
    built and its sinogram placed through the public entry
    (``prepare_sino_for_devices``), with no pin of either kind, so the settle
    inside it is the same automatic choice a user's first reconstruction
    makes.  The expected count comes from the floors table alone -- the widest
    count ``_widening_floors.admitted`` accepts at this cell's size -- which is
    an independent reading of the same table the policy consults, under the
    assumption that memory is ample.  On the nightly's dedicated node that
    assumption holds by a wide margin at this cell, and a capacity refusal
    would itself be an anomaly; the recorded per-count reasons say which rule
    drove any mismatch.

    ``ok`` is False when the realized count differs from the expected one,
    when the layout did not come from the automatic branch, when the guard was
    disabled in the environment, or when a device-count pin had leaked into
    this process (popped and recorded, since a leaked pin silently un-tests
    exactly the path this check exists to cover).
    """
    import torch
    import mbirtorch  # noqa: F401
    from mbirtorch import _widening_floors as wf

    leaked_pin = os.environ.pop("MBIRTORCH_NUM_DEVICES", None)
    size = tuple(AUTO_CHOICE_SIZE)
    geometry = AUTO_CHOICE_GEOMETRY
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    result = dict(kind="auto_choice", geometry=geometry, size=list(size),
                  visible_devices=int(visible),
                  leaked_env_pin=leaked_pin,
                  guard_enabled=bool(wf.guard_enabled()),
                  floors_stale_note=wf.stale_note())

    elements = wf.sinogram_elements(size)
    admitted = {}
    for n in range(1, max(int(visible), 1) + 1):
        ok, why = wf.admitted(geometry, n, elements)
        admitted[int(n)] = {"admitted": bool(ok), "why": str(why)}
    result["admitted_by_count"] = admitted
    expected = max([n for n, v in admitted.items() if v["admitted"]] or [1])
    result["expected_n_devices"] = int(expected)

    model = make_model(config, geometry, size, platform_key)
    sino_np = pt.make_sinogram(config, size)
    model.prepare_sino_for_devices(sino_np)
    realized = int(model.sino_placement.n_devices)
    result["realized_n_devices"] = realized
    result["layout_is_automatic"] = bool(
        getattr(model, "device_layout_is_automatic", False))
    result["choice_rejections"] = [
        [int(count), str(why)] for count, why
        in (getattr(model, "device_choice_rejections", None) or [])]

    problems = []
    if leaked_pin is not None:
        problems.append(f"MBIRTORCH_NUM_DEVICES={leaked_pin!r} had leaked into "
                        f"this process (popped before the settle)")
    if not result["guard_enabled"]:
        problems.append("the widening guard is disabled in this environment, "
                        "so the floors were never consulted")
    if not result["layout_is_automatic"]:
        problems.append("the settled layout is not marked automatic, so "
                        "something pinned or configured it")
    if realized != expected:
        problems.append(f"the automatic choice took {realized} device(s) where "
                        f"the shipped floors say {expected}")
    result["problems"] = problems
    result["ok"] = not problems
    return result


def run_worker(argv):
    p = argparse.ArgumentParser(description="torch_backend_writer worker (internal)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=["setup", "measure", "auto-choice"], required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--platform", required=True)
    p.add_argument("--geometry", default=None)
    p.add_argument("--op", default=None)
    p.add_argument("--size", default=None)
    p.add_argument("--device-counts", type=int, nargs="+", default=None)
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    assert_platform(a.platform)
    if a.mode == "setup":
        sc.write_worker_result(a.out_file, probe_environment(a.platform))
        return
    config = pt.Config.from_dict(sc.load_yaml(a.config))
    if a.mode == "auto-choice":
        sc.write_worker_result(a.out_file, auto_choice_check(config, a.platform))
        return
    res = measure_cell_group(config, a.geometry, a.op, a.size, a.device_counts,
                             a.platform, a.out_file)
    sc.write_worker_result(a.out_file, res)


# ── Environment identity (the platform-mismatch guard class) ──────────────────
def probe_environment(platform_key):
    """What this run measured ON, recorded so a night-to-night shift can be attributed
    to the environment rather than the code.  Runs in a worker, since it imports torch.
    """
    import torch
    import mbirtorch
    info = {
        "platform": platform_key,
        "device_label": (f"GPU-TORCH ({torch.cuda.get_device_name(0)})"
                         if torch.cuda.is_available()
                         else f"CPU-TORCH ({_platform.processor() or _platform.machine()})"),
        "max_devices": int(torch.cuda.device_count()) if torch.cuda.is_available() else 1,
        "toolchain": {
            "torch": str(torch.__version__),        # TorchVersion is a str SUBCLASS; yaml.safe_dump refuses it
            "torch_cuda": str(torch.version.cuda) if getattr(torch.version, "cuda", None) else None,
            "python": _platform.python_version(),
            "executable": sys.executable,
            "loaded_modules": os.environ.get("LOADEDMODULES"),
        },
        "packages": sc.installed_packages(),
        "mbirtorch_version": str(mbirtorch.__version__),
    }
    try:
        import triton
        info["toolchain"]["triton"] = str(triton.__version__)
    except Exception:                              # noqa: BLE001
        info["toolchain"]["triton"] = None
    # Which projector bodies this run will actually use.  The nightly measures the
    # SHIPPED configuration (kernels default-on), so this is recorded, not forced —
    # the arm-check discipline of phase5_findings.md applied to the nightly.
    try:
        from mbirtorch import kernel_availability as ka
        ok, reason = ka.triton_available()
        info["kernels"] = {"triton_available": bool(ok), "reason": str(reason),
                           "disable_env": os.environ.get("MBIRTORCH_DISABLE_TRITON")}
    except Exception as e:                         # noqa: BLE001
        info["kernels"] = {"triton_available": None, "reason": f"probe failed: {e}"}
    return info


def git_provenance(root):
    """{git_commit, git_commit_date, git_branch, ...} for the mbirtorch checkout."""
    def _g(args):
        try:
            r = subprocess.run(["git", "-C", root, *args], capture_output=True,
                               text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:                          # noqa: BLE001
            return None
    dirty = _g(["status", "--porcelain"]) or ""
    dirty_files = [ln[3:].split(" -> ")[-1] for ln in dirty.splitlines()]
    return {"git_commit": _g(["rev-parse", "HEAD"]),
            "git_commit_date": _g(["show", "-s", "--format=%cI", "HEAD"]),
            "git_branch": _g(["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_dirty": bool(dirty),
            "git_dirty_files": dirty_files[:20],
            "git_dirty_code": any(f.startswith("mbirtorch/") for f in dirty_files)}


# ── Orchestrator ──────────────────────────────────────────────────────────────
def run(config, platform_key):
    """Sweep, gate, and write the dated YAML + companions."""
    script = os.path.abspath(__file__)
    os.makedirs(config.out_dir, exist_ok=True)
    worker_env = {"PYTHONPATH": os.pathsep.join(
        [p for p in (config.lib_root, os.environ.get("PYTHONPATH")) if p])}

    print("=" * 72)
    print("  torch_backend_writer — mbirtorch regression sweep")
    print(f"  lib_root (under test): {config.lib_root}")
    print(f"  out_dir:               {config.out_dir}")
    print(f"  platform / date / tag: {platform_key} / {config.date} / {config.run_tag or '-'}")
    print("=" * 72)

    setup, rc = sc.run_worker(script, ["--worker", "--mode", "setup",
                                       "--platform", platform_key], extra_env=worker_env)
    if setup is None:
        print(f"  ERROR: setup worker produced no result (rc={rc}); aborting.")
        return None
    max_dev = int(setup.get("max_devices") or 1)
    print(f"  device: {setup['device_label']}   visible devices: {max_dev}")
    print(f"  torch {setup['toolchain']['torch']} · kernels: {setup.get('kernels')}")

    device_counts = [n for n in config.device_counts if n <= max_dev]
    if not device_counts:
        print(f"  ERROR: no requested device count fits {max_dev} visible device(s); aborting.")
        return None

    fd, cfg_path = tempfile.mkstemp(suffix=".yaml", prefix="torch_cfg_")
    os.close(fd)
    sc.save_yaml(cfg_path, config.to_dict())

    # The automatic-device-choice check runs once, before the sweep, wherever a
    # choice exists (two or more visible devices).  Every measured row below
    # pins its count, so this is the only place the automatic path runs.  The
    # worker gets NO device-count pin.
    auto_choice = None
    if platform_key == "gpu-torch" and max_dev >= 2:
        print("\n=== automatic device choice (unpinned settle, "
              f"{AUTO_CHOICE_GEOMETRY} {sc.size_label(AUTO_CHOICE_SIZE)}) ===")
        auto_choice, _rc = sc.run_worker(
            script, ["--worker", "--mode", "auto-choice", "--config", cfg_path,
                     "--platform", platform_key], extra_env=worker_env)
        if auto_choice is None:
            auto_choice = {"kind": "auto_choice", "ok": False,
                           "problems": ["the auto-choice worker produced no "
                                        "result"]}
        if auto_choice.get("ok"):
            print(f"  ok: chose {auto_choice.get('realized_n_devices')} "
                  f"device(s), as the shipped floors say "
                  f"(expected {auto_choice.get('expected_n_devices')}, "
                  f"{auto_choice.get('visible_devices')} visible)")
        else:
            print("  AUTO-CHOICE MISMATCH:")
            for problem in auto_choice.get("problems") or []:
                print(f"    {problem}")
            for count, why in auto_choice.get("choice_rejections") or []:
                print(f"    count {count} rejected: {why}")
        if auto_choice.get("floors_stale_note"):
            print(f"  note: {auto_choice['floors_stale_note']}")
    else:
        auto_choice = {"kind": "auto_choice",
                       "skipped": ("single-device night: no choice exists"
                                   if platform_key == "gpu-torch" else
                                   "cpu platform: the automatic choice is a "
                                   "CUDA path")}
        print(f"\n  automatic device choice check skipped: "
              f"{auto_choice['skipped']}")

    cells = []
    swept_counts = set()
    for geometry in config.geometries:
        gs = (config.geom_sizes.get(geometry, {}) or {}).get(platform_key) \
            or config.sizes[platform_key]
        size_labels = [sc.size_label(s) for s in gs]
        for op in (config.geom_ops.get(geometry) or config.ops):
            for label in size_labels:
                # Per-(geometry, size) counts: only the MULTI_DEVICE_SIZE_LABELS cells
                # sweep n>1; everything else, and the whole denoiser, stays n=1.
                gdc = cell_device_counts(geometry, label, device_counts)
                swept_counts.update(gdc)
                print(f"\n=== {geometry} | {op} | {label} @ n={gdc} ===")
                # Second, independent pin layer (nightly_plan.md §3(d)): the process-wide
                # env pin covers any model a code path constructs WITHOUT an explicit
                # configure_devices call (explicit pins always win over it).  It is a
                # single value per process, so it is exportable only when this worker
                # sweeps exactly one count — true for every n=1 row today.  The n>1
                # increment sweeps several counts per worker and relies on the explicit
                # pin + realized-list assertion alone.
                cell_env = (dict(worker_env, MBIRTORCH_NUM_DEVICES=str(gdc[0]))
                            if len(gdc) == 1 else worker_env)
                args = ["--worker", "--mode", "measure", "--config", cfg_path,
                        "--platform", platform_key, "--geometry", geometry, "--op", op,
                        "--size", label, "--device-counts", *[str(n) for n in gdc]]
                res, _rc = sc.run_worker(script, args, extra_env=cell_env)
                if not res:
                    print(f"  (no result for {geometry}/{op}/{label})")
                    continue
                rows = res.get("rows") or []
                sc.annotate_speedups(rows)
                cells.extend(rows)
                for f in (res.get("failures") or []):
                    cells.append({"geometry": geometry, "op": op, "size": label,
                                  "n_devices": f["n_devices"], "failed": True,
                                  "oom": bool(f.get("oom")), "error": f.get("error")})
    os.path.exists(cfg_path) and os.remove(cfg_path)

    prov = git_provenance(config.lib_root)
    if prov.get("git_branch") in (None, "", "HEAD") and config.run_tag:
        prov["git_branch"] = config.run_tag
    file_tag = pt._file_tag(prov, config.date)   # COMMIT-time tag: one file per commit,
    #                                              sorts chronologically, overwrites on re-measure

    records_path = os.path.join(config.out_dir, f"records_{platform_key}.yaml")
    records = (sc.load_yaml(records_path) or {}) if os.path.exists(records_path) else {}
    new_lines, n_baselines = pt.update_records(records, cells, prov.get("git_commit") or "?",
                                               config.date)
    sc.save_yaml(records_path, records)

    cfg_dict = config.to_dict()
    cfg_dict["backend"] = "torch"
    result = {
        "kind": "regression", "date": config.date, "platform": platform_key,
        # mbirtorch has no per-geometry sharding capability probe: every geometry either
        # supports placement or, for the denoiser, is deliberately held at one device.
        "sharding_by_geom": {g: (g != "denoiser") for g in config.geometries},
        "device_label": setup["device_label"], **prov,
        "mbirjax_version": f"mbirtorch {setup['mbirtorch_version']}",
        "toolchain": setup["toolchain"],
        "packages": setup.get("packages") or {},
        "kernels": setup.get("kernels"),
        "mem_kind": "gpu_peak_per_device" if platform_key == "gpu-torch" else "cpu_rss",
        "dep_gen": 0, "run_reason": "commit", "jax_available": None,
        "measured_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": cfg_dict, "device_counts": sorted(swept_counts), "cells": cells,
        "policy": {},
        "auto_choice": auto_choice,
    }

    gate_dict = None
    if config.compare_to_prior:
        W = max(1, int(getattr(config, "mem_gate_window", 1)))
        priors = pt._find_priors(config.out_dir, platform_key, file_tag, W)
        if priors:
            ref = sc.load_yaml(priors[0]) or {}
            gate_result, gate_ref = ((result, ref) if W <= 1
                                     else pt._apply_mem_window(result, ref, priors, W))
            refs = [(f"prior:{os.path.basename(priors[0])}", gate_ref)]
            gate_dict = pt.gate_run(gate_result, refs, config)
        else:
            gate_dict = pt.gate_run(result, [], config)   # cold start -> all-SOFT
        result["gate"] = gate_dict

    # A failed auto-choice check gates HARD, cold start included: its
    # expectation comes from the shipped floors table, not from a prior run,
    # so there is nothing to warm up.  A skipped check gates nothing.
    if auto_choice and not auto_choice.get("skipped") and not auto_choice.get("ok"):
        line = ("[auto-choice] the automatic device choice disagrees with the "
                "shipped floors: " + "; ".join(auto_choice.get("problems")
                                               or ["no detail recorded"]))
        if gate_dict is None:
            gate_dict = {"result": "fail", "hard": [line], "soft": [],
                         "compared_to": []}
        else:
            gate_dict["hard"] = list(gate_dict.get("hard") or []) + [line]
            gate_dict["result"] = "fail"
        result["gate"] = gate_dict

    out_path = os.path.join(config.out_dir, f"regression_{platform_key}_{file_tag}.yaml")
    sc.save_yaml(out_path, result)
    try:
        import regression_to_table
        regression_to_table.write_table(regression_to_table.load_yaml(out_path),
                                        os.path.splitext(out_path)[0] + "_table.yaml")
    except Exception as e:                         # noqa: BLE001
        print(f"[warn] companion _table.yaml not written: {e}")

    pt._print_summary(cells)
    if new_lines:
        print(f"\n  {len(new_lines)} NEW RECORD(S) this run:")
        for line in new_lines:
            print(line)
    elif n_baselines:
        print(f"\n  established {n_baselines} baseline record(s) (first run for these cells)")
    if gate_dict:
        pt._print_gate(gate_dict)
    print(f"\nOutput written to: {out_path}")
    print(f"Record book:       {records_path}")
    return result


def write_tests_log(lib_root, out_dir, platform_key, date):
    """Run the mbirtorch suite and capture it beside the run, as the jax nightly does.

    The suite is the torch series' cross-framework coverage: test_vs_goldens.py carries
    the torch-vs-jax value check, which is why this plan puts no cross-framework column
    in the nightly itself.
    """
    tests_path = os.path.join(out_dir, f"tests_{platform_key}_{date}.txt")
    nproc = "4" if platform_key == "gpu-torch" else "8"
    runner = os.path.join(lib_root, "dev_scripts", "run_tests.sh")
    env = {**os.environ, "PYTEST_NPROC": nproc}
    if os.path.isfile(runner):
        # run_tests.sh uses a path RELATIVE to dev_scripts/, so it must run from there.
        proc = subprocess.run(["bash", "run_tests.sh"], cwd=os.path.dirname(runner),
                              capture_output=True, text=True, env=env)
    else:
        proc = subprocess.run([sys.executable, "-m", "pytest", "tests", "-ra", "-n", nproc],
                              cwd=lib_root, capture_output=True, text=True, env=env)
    with open(tests_path, "w") as f:
        f.write(proc.stdout + proc.stderr)
    print(f"wrote {tests_path}")
    return tests_path


def main():
    platform_key = os.environ.get("REG_TORCH_PLATFORM")
    lib_root = os.environ.get("REG_TORCH_LIB_ROOT")
    out_dir = os.environ.get("REG_TORCH_OUT_DIR")
    for name, val in (("REG_TORCH_PLATFORM", platform_key),
                      ("REG_TORCH_LIB_ROOT", lib_root),
                      ("REG_TORCH_OUT_DIR", out_dir)):
        if not val:
            raise SystemExit(f"torch_backend_writer: required env var {name} is not set")
    assert_no_calibration()
    date = os.environ.get("REG_TORCH_DATE") or datetime.datetime.now().strftime("%Y%m%d")
    counts = [int(x) for x in (os.environ.get("REG_TORCH_DEVICE_COUNTS") or "1").split()]
    config = build_config(platform_key, out_dir, date,
                          os.environ.get("REG_TORCH_RUN_TAG", ""), lib_root, counts,
                          gate=os.environ.get("REG_TORCH_GATE", "1") == "1")
    if os.environ.get("REG_TORCH_SMOKE") == "1":
        # Fast plumbing check (NOT a measurement): one tiny cell, end to end.
        config.geometries = ["parallel"]
        config.ops = ["back"]
        config.sizes = {platform_key: [[40, 40, 48]]}
        config.geom_sizes = {}
        config.device_counts = [1]

    result = run(config, platform_key)
    if result is None:
        raise SystemExit(2)
    # The nightly wrapper owns the test step (live output, crash detection, alert mail), so it
    # exports REG_TORCH_SKIP_TESTS=1; a standalone/manual invocation still runs the suite here.
    if os.environ.get("REG_TORCH_SMOKE") != "1" and os.environ.get("REG_TORCH_SKIP_TESTS") != "1":
        write_tests_log(lib_root, out_dir, platform_key, date)
    if config.gate and (result.get("gate") or {}).get("result") == "fail":
        raise SystemExit(1)      # HARD regression -> the wrapper turns this into an alert


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
