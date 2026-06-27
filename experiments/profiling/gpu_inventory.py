"""
experiments/profiling/gpu_inventory.py
────────────────────────────────────────
Step 0 of the GPU phase: a cheap, self-contained probe to run on Gautschi the FIRST time you
have an H100 allocation, so we know — from data, not guesses — what's available before planning
the heavyweight steps (and whether anything needs installing, which is gated on your OK).

Run (on the cluster, in the fresh GPU env):
    python experiments/profiling/gpu_inventory.py

It prints, and writes results/gpu_inventory.yaml (readable over the Samba mount), the things the
next steps hinge on:
  * JAX backend + visible H100s + device_kind (does jax see the GPUs at all)
  * jax / jaxlib versions (must match the Mac's 0.10.2 for apples-to-apples, or note the delta)
  * nsys / ncu / nvidia-smi on PATH + versions  (are the NVIDIA tools already there, e.g. via a module)
  * tensorboard + tensorboard-plugin-profile importable  (optional — Perfetto needs neither)
  * GPU topology (nvidia-smi -L / topo -m) and an idle clock/temp sample  (the throttle/NUMA
    pre-flight from mbirjax/.claude/lessons.md — the warmest-at-idle card is the one that throttles)
"""
import os
import sys
import shutil
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "tooling", "scaling_tests")))

import mbirjax            # noqa: E402,F401  device-setup-first
import jax                # noqa: E402
import scaling_common as sc   # noqa: E402  (reuse gpu_topology / sample_gpu_health / YAML)


def _tool(name):
    """{path, version} for a CLI tool on PATH, or {present: False}."""
    path = shutil.which(name)
    if not path:
        return {"present": False}
    ver = None
    for flag in ("--version", "-v"):
        try:
            r = subprocess.run([name, flag], capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                ver = (r.stdout or r.stderr).strip().splitlines()[0]
                break
        except Exception:   # noqa: BLE001
            pass
    return {"present": True, "path": path, "version": ver}


def _importable(mod):
    try:
        __import__(mod)
        return True
    except Exception:   # noqa: BLE001
        return False


def main():
    try:
        import jaxlib
        jaxlib_ver = jaxlib.__version__
    except Exception:   # noqa: BLE001
        jaxlib_ver = None

    backend = jax.default_backend()
    gpus = []
    try:
        gpus = jax.devices("gpu")
    except Exception:   # noqa: BLE001
        pass

    inv = {
        "backend": backend,
        "n_gpus": len(gpus),
        "device_kind": gpus[0].device_kind if gpus else None,
        "jax": jax.__version__,
        "jaxlib": jaxlib_ver,
        "mac_reference_jax": "0.10.2",   # keep the GPU env on this version for apples-to-apples
        "tools": {name: _tool(name) for name in ("nsys", "ncu", "nvidia-smi")},
        "tensorboard": _importable("tensorboard"),
        "tensorboard_plugin_profile": _importable("tensorboard_plugin_profile"),
        "topology": sc.gpu_topology(),
        "gpu_health_idle": sc.sample_gpu_health(),
    }

    print("=" * 78)
    print("  GPU INVENTORY")
    print("=" * 78)
    print(f"  backend={inv['backend']}  n_gpus={inv['n_gpus']}  kind={inv['device_kind']}")
    print(f"  jax={inv['jax']}  jaxlib={inv['jaxlib']}  (Mac reference jax=0.10.2)")
    for name, d in inv["tools"].items():
        print(f"  {name:<11}: " + ("present " + (d.get('version') or '') if d["present"] else "MISSING"))
    print(f"  tensorboard={inv['tensorboard']}  tensorboard_plugin_profile={inv['tensorboard_plugin_profile']}")
    health = inv["gpu_health_idle"]
    if health:
        print("  idle GPU health (warmest-at-idle card is the throttle suspect):")
        for g in health:
            print(f"    GPU{g.get('index')}  {g.get('temp_c')}C  sm={g.get('sm_mhz')}MHz")
    out = os.path.join(sc.RESULTS_DIR, "gpu_inventory.yaml")
    sc.save_yaml(out, inv)


if __name__ == "__main__":
    main()
