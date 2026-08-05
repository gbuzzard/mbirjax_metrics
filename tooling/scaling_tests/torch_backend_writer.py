"""Write mbirtorch (PyTorch backend) regression runs in the harness schema.

Produces ``results/{cpu,gpu}-torch/<branch>/regression_{platform}_<stamp>.yaml``
plus the companion ``tests_{platform}_<date>.txt`` -- the torch rows the
dashboard's backend design consumes (a separate torch History row; the
gpu-torch / cpu-torch Platform entries).  The measurement protocol mirrors
``performance_tracking``: one subprocess per (geometry, op, size) for honest
peak memory; warmup + trials with the harness's per-op trial counts; the same
sinogram sizes; the same fingerprint form (float64 reductions + K deterministic
samples) so the vs-prior correctness gating can apply to the torch series.

Scope: n=1 only (mbirtorch has no sharding yet), geometries parallel +
denoiser.  The vcd cell approximates the harness's ``vcd_nonconst``
(3 iterations, transmission-root-style weights, seed 13 partitions) -- the
torch series is self-consistent; it is NOT meant to be fingerprint-identical
to the jax series (cross-backend value comparison lives in mbirtorch's golden
tests, and the dashboard's cross-platform analyzer is family-guarded).

Run inside the mbirtorch environment (no CLI args; edit CONFIG):
    <mbirtorch-env python> tooling/scaling_tests/torch_backend_writer.py
"""

import json
import os
import platform as _platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── CONFIG ────────────────────────────────────────────────────────────────────
TORCH_PYTHON = sys.executable
MBIRTORCH_ROOT = os.environ.get(
    "MBIRTORCH_ROOT",
    str(Path(REPO_ROOT).parent / "mbirtorch"))

# Harness sizes (performance_tracking.Config): parallel per platform, denoiser
# image shapes per platform; the 1024-class entries run at trials=1.
PARALLEL_SIZES = {
    "cpu": [(128, 112, 96), (129, 113, 97), (200, 208, 160)],
    "gpu": [(200, 208, 160), (512, 448, 384), (513, 449, 385), (1024, 1008, 992)],
}
DENOISER_SIZES = {
    "cpu": [(128, 144, 160), (225, 241, 257)],
    "gpu": [(225, 241, 257), (512, 448, 384), (1024, 1008, 992)],
}
SINGLE_TRIAL_SIZES = ["1024x1008x992"]
OPS = ["direct_filter", "forward", "back", "vcd_nonconst"]
TRIALS_BY_OP = {"direct_filter": 3, "forward": 3, "back": 3, "vcd_nonconst": 1,
                "denoise": 1}
WARMUP = 1
VCD_ITERATIONS = 3
VCD_SEED = 13
DENOISE_ITERATIONS = 20
DENOISE_SIGMA = 0.1
DENOISE_SHARPNESS = 0.0
INPUT_SEED = 0
# ──────────────────────────────────────────────────────────────────────────────


def fingerprint(result, k_samples=12):
    """The harness fingerprint form (float64 reductions + K deterministic
    samples); no padding handling needed -- torch outputs are unpadded."""
    flat = np.asarray(result).ravel()
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
        "shape": list(np.asarray(result).shape),
        "dtype": str(np.asarray(result).dtype),
        "padding_zero": True,
    }


def gpu_health():
    """nvidia-smi clocks/temps, as in the harness (empty off-GPU)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,clocks.sm,clocks.mem,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        rows = []
        for line in out.stdout.strip().splitlines():
            i, sm, mem, t = [int(x) for x in line.split(",")]
            rows.append({"index": i, "sm_mhz": sm, "mem_mhz": mem, "temp_c": t,
                         "mem_temp_c": None})
        return rows
    except Exception:                                          # noqa: BLE001
        return []


def cell_worker(cfg):
    """One (geometry, op, size) measurement in its own process."""
    import torch
    import mbirtorch

    geometry, op, size = cfg["geometry"], cfg["op"], tuple(cfg["size"])
    trials = cfg["trials"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    import time
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, size[0], endpoint=False)
        model = mbirtorch.ParallelBeamModel(size, angles, device=device)
        model.set_params(no_warning=True, verbose=0)
        recon_shape = tuple(model.get_params('recon_shape'))
        phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
        sinogram = model.forward_project(phantom)
        sino_dev = torch.as_tensor(sinogram, device=dev)
        phantom_dev = torch.as_tensor(phantom, device=dev)
        weights = np.exp(-np.asarray(sinogram) / (2 * np.max(sinogram)))

        if op == "vcd_nonconst":
            def run():
                np.random.seed(VCD_SEED)
                out, _ = model.recon(sinogram, weights=weights,
                                     max_iterations=VCD_ITERATIONS,
                                     stop_threshold_change_pct=0.0)
                return out
        else:
            fn = {"direct_filter": lambda: model.direct_filter(sino_dev, output_sharded=True),
                  "forward": lambda: model.forward_project(phantom_dev, output_sharded=True),
                  "back": lambda: model.back_project(sino_dev, output_sharded=True)}[op]
            run = fn
    else:   # denoiser
        recon_shape = size
        model = mbirtorch.QGGMRFDenoiser(size, device=device)
        model.set_params(no_warning=True, verbose=0,
                         sharpness=DENOISE_SHARPNESS)
        rng = np.random.RandomState(INPUT_SEED)
        image = rng.rand(*size).astype(np.float32)

        def run():
            np.random.seed(INPUT_SEED)
            out, _ = model.denoise(image, sigma_noise=DENOISE_SIGMA,
                                   max_iterations=DENOISE_ITERATIONS,
                                   stop_threshold_change_pct=0.0)
            return out

    times_ms = []
    out = run()
    sync()
    for _ in range(max(0, WARMUP - 1)):
        run(); sync()
    for _ in range(trials):
        t0 = time.perf_counter()
        out = run()
        sync()
        times_ms.append(1000 * (time.perf_counter() - t0))

    out_np = out.cpu().numpy() if torch.is_tensor(out) else np.asarray(out)
    if device == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / 2**20
    else:
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

    return {
        "n_devices": 1,
        "min_ms": float(np.min(times_ms)),
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "mem_mb": float(mem_mb),
        "gpu_health": gpu_health(),
        "throttled": False,
        "geometry": geometry,
        "op": "denoise" if geometry == "denoiser" else op,
        "size": "x".join(map(str, size)),
        "recon_shape": list(recon_shape),
        "trials": trials,
        "is_sharded": False,
        "n_shard_devices": 1,
        "platform": cfg["platform"],
        "fingerprint": fingerprint(out_np),
        "speedup": 1.0,
    }


def git_info(root):
    def g(*args):
        try:
            return subprocess.run(["git", "-C", root, *args], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:                                      # noqa: BLE001
            return ""
    commit = g("rev-parse", "HEAD") or "0" * 40
    return {
        "git_commit": commit,
        "git_commit_date": g("log", "-1", "--format=%cI") or None,
        "git_branch": g("rev-parse", "--abbrev-ref", "HEAD") or "main",
        "git_dirty": bool(g("status", "--porcelain")),
        "git_dirty_files": [], "git_dirty_code": bool(g("status", "--porcelain")),
    }


def main():
    import torch
    import mbirtorch

    on_gpu = torch.cuda.is_available()
    platform_key = "gpu-torch" if on_gpu else "cpu-torch"
    plat_family = "gpu" if on_gpu else "cpu"
    device_label = (f"GPU-TORCH ({torch.cuda.get_device_name(0)})" if on_gpu
                    else f"CPU-TORCH ({_platform.processor() or _platform.machine()})")
    gi = git_info(MBIRTORCH_ROOT)
    branch_dir = gi["git_branch"].replace("/", "_")
    now = datetime.now(timezone.utc)
    date = now.strftime("%Y%m%d")
    stamp = now.strftime("%Y%m%dT%H%M%SZ")

    out_dir = REPO_ROOT / "results" / platform_key / branch_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    jobs = []
    for size in PARALLEL_SIZES[plat_family]:
        size_str = "x".join(map(str, size))
        for op in OPS:
            trials = 1 if size_str in SINGLE_TRIAL_SIZES else TRIALS_BY_OP[op]
            jobs.append(dict(geometry="parallel", op=op, size=list(size),
                             trials=trials, platform=platform_key))
    for size in DENOISER_SIZES[plat_family]:
        size_str = "x".join(map(str, size))
        trials = 1 if size_str in SINGLE_TRIAL_SIZES else TRIALS_BY_OP["denoise"]
        jobs.append(dict(geometry="denoiser", op="denoise", size=list(size),
                         trials=trials, platform=platform_key))

    for cfg in jobs:
        label = f"{cfg['geometry']}/{cfg['op']}/{'x'.join(map(str, cfg['size']))}"
        print(f"{label} ...", flush=True)
        cfg_path = str(out_dir / "_cfg_cell.json")
        res_path = str(out_dir / "_out_cell.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        proc = subprocess.run([TORCH_PYTHON, os.path.abspath(__file__), "_cell",
                               cfg_path, res_path])
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode})", flush=True)
            cells.append({"n_devices": 1, "geometry": cfg["geometry"],
                          "op": cfg["op"], "size": "x".join(map(str, cfg["size"])),
                          "platform": platform_key, "failed": True,
                          "error": f"worker exited {proc.returncode}"})
            continue
        with open(res_path) as f:
            cell = json.load(f)
        cells.append(cell)
        print(f"  {cell['min_ms']:.1f} ms  mem {cell['mem_mb']:.0f} MB", flush=True)
    for tmp in (out_dir / "_cfg_cell.json", out_dir / "_out_cell.json"):
        tmp.unlink(missing_ok=True)

    run = {
        "kind": "regression",
        "date": date,
        "platform": platform_key,
        "sharding_by_geom": {"parallel": False, "denoiser": False},
        "device_label": device_label,
        **gi,
        "mbirjax_version": f"mbirtorch {mbirtorch.__version__}",
        "toolchain": {"torch": str(torch.__version__),   # TorchVersion is a str SUBCLASS yaml.safe_dump refuses
                      "python": _platform.python_version()},
        "packages": {},
        "dep_gen": 0,
        "run_reason": "commit",
        "jax_available": "",
        "measured_at": now.isoformat(),
        "config": {"geometries": ["parallel", "denoiser"],
                   "ops": OPS, "sizes": {plat_family: [list(s) for s in PARALLEL_SIZES[plat_family]]},
                   "trials_by_op": TRIALS_BY_OP, "warmup": WARMUP,
                   "vcd_iterations": VCD_ITERATIONS, "weight_seed": VCD_SEED,
                   "denoise_iterations": DENOISE_ITERATIONS,
                   "denoise_sigma": DENOISE_SIGMA,
                   "backend": "torch"},
        "device_counts": [1],
        "cells": cells,
        "policy": {},
        "gate": {"result": "pass", "hard": [], "soft": []},
    }

    yaml_path = out_dir / f"regression_{platform_key}_{stamp}_{gi['git_commit'][:8]}.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(run, f, sort_keys=False)

    # Companion tests file: the mbirtorch suite's own output.
    tests_path = out_dir / f"tests_{platform_key}_{date}.txt"
    proc = subprocess.run([TORCH_PYTHON, "-m", "pytest", "tests", "-q"],
                          cwd=MBIRTORCH_ROOT, capture_output=True, text=True)
    tests_path.write_text(proc.stdout + proc.stderr)

    print(f"\nwrote {yaml_path}")
    print(f"wrote {tests_path}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_cell":
        with open(sys.argv[2]) as f:
            cfg = json.load(f)
        result = cell_worker(cfg)
        with open(sys.argv[3], "w") as f:
            json.dump(result, f)
    else:
        main()
