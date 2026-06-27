"""
experiments/profiling/profiling_config.py
────────────────────────────────────────────
Single source of run config for the profiling scripts — parsed from ``profiling.env`` (KEY=VALUE,
like the regression's run_configs.env), so the scripts ``from profiling_config import ...`` instead
of each carrying their own CONFIG block.

Precedence: a same-named ENVIRONMENT VARIABLE overrides the .env line, so a one-off needs no edit:
    SIZE=512x512x512 python experiments/profiling/trace_back_projection.py

Importing this module also sets ``MBIRJAX_NUM_CPU_DEVICES`` (device-setup-first) to the max device
count any script needs — so a script just imports profiling_config BEFORE mbirjax and the virtual
CPU mesh is sized correctly.  This module is JAX-free (pure parsing).

Sizes are ``LxWxH`` labels (kept ASYMMETRIC on purpose — symmetric sizes can mask axis/stride
effects); lists are comma-separated.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV = os.path.join(_HERE, "profiling.env")


def _load_env():
    cfg = {}
    if os.path.exists(_ENV):
        with open(_ENV) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip()
    cfg.update({k: os.environ[k] for k in list(cfg) if k in os.environ})   # env overrides the file
    return cfg


_C = _load_env()


def _get(key, default):
    return _C.get(key, default)


def _size(s):
    return tuple(int(x) for x in s.lower().split("x"))


def _ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _strs(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _sizes(s):
    return [_size(x) for x in s.split(",") if x.strip()]


GEOMETRY       = _get("GEOMETRY", "cone")
OPS            = _strs(_get("OPS", "forward,back"))
# Per-platform sizes (CPU can't run the big GPU sizes; no cross-platform cell needed here).  Resolve
# with size_for(platform) / sizes_for(platform) AFTER the script knows its platform (post `import jax`),
# since this module is JAX-free and can't detect the device itself.
SIZE_CPU       = _size(_get("SIZE_CPU", "200x208x160"))
SIZE_GPU       = _size(_get("SIZE_GPU", "512x448x384"))
SIZES_CPU      = _sizes(_get("SIZES_CPU", "128x112x96,200x208x160"))
SIZES_GPU      = _sizes(_get("SIZES_GPU", "200x208x160,512x448x384"))
_SIZE_OVR      = _get("SIZE", "").strip()    # un-suffixed SIZE/SIZES = one-off override for BOTH platforms
_SIZES_OVR     = _get("SIZES", "").strip()
N_DEVICES      = int(_get("N_DEVICES", "1"))
N_DEVICES_LIST = _ints(_get("N_DEVICES_LIST", "1,2"))
WARMUP         = int(_get("WARMUP", "2"))
TRACE_ITERS    = int(_get("TRACE_ITERS", "3"))
STATIC_TRIALS  = int(_get("STATIC_TRIALS", "3"))
COMPILE_TRIALS = int(_get("COMPILE_TRIALS", "3"))
PROFILE_CALLS  = int(_get("PROFILE_CALLS", "2"))
TOP_N          = int(_get("TOP_N", "30"))

# Device-setup-first: size the CPU virtual-device mesh before any `import mbirjax`.  setdefault
# respects a value already set by the shell/cluster.
os.environ.setdefault("MBIRJAX_NUM_CPU_DEVICES", str(max([N_DEVICES, *N_DEVICES_LIST])))


def size_for(platform):
    """The single problem size for this platform — SIZE_GPU on gpu, SIZE_CPU otherwise; an
    un-suffixed SIZE override (env/`.env`) wins for both."""
    if _SIZE_OVR:
        return _size(_SIZE_OVR)
    return SIZE_GPU if platform == "gpu" else SIZE_CPU


def sizes_for(platform):
    """The size SWEEP for this platform (static/compile scripts) — SIZES_GPU on gpu else SIZES_CPU;
    an un-suffixed SIZES override wins for both."""
    if _SIZES_OVR:
        return _sizes(_SIZES_OVR)
    return SIZES_GPU if platform == "gpu" else SIZES_CPU
