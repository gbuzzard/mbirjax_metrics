"""
experiments/profiling/cuda_profiler.py
────────────────────────────────────────
Tiny ctypes wrapper around cudaProfilerStart/Stop so an `ncu` run can be scoped to a code REGION.

Why: `ncu` instruments the WHOLE process and serializes kernels, so when it wraps a JAX program the
8 min wall time is dominated by `ncu` sitting on top of JAX's import + compile/autotuning during
warmup — not the roofline collection.  Bracketing the warm calls with cudaProfilerStart/Stop and
running `ncu --profile-from-start off` makes `ncu` ignore everything UNTIL the region, cutting these
runs to well under a minute.

JAX doesn't expose the CUDA profiler API, so we call libcudart directly.  No-op (warns once) when
libcudart isn't loadable (CPU / no CUDA) — so the ncu scripts still run for a local smoke test.
"""
import ctypes
import ctypes.util
from contextlib import contextmanager


def _load_cudart():
    """Best-effort load of libcudart across naming conventions (find_library, then common sonames)."""
    candidates = [ctypes.util.find_library("cudart"),
                  "libcudart.so", "libcudart.so.12", "libcudart.so.11.0"]
    for name in candidates:
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


_CUDART = _load_cudart()
_warned = False


@contextmanager
def profiler_range():
    """Bracket a region with cudaProfilerStart/Stop.

    With `ncu --profile-from-start off` (or `nsys --capture-range=cudaProfilerApi`), the tool profiles
    ONLY what runs inside this `with`.  Put model build + warmup OUTSIDE it.  Block the device before
    leaving so the profiled kernels actually complete inside the range.  A no-op when there is no CUDA.
    """
    global _warned
    started = False
    if _CUDART is not None:
        try:
            started = (_CUDART.cudaProfilerStart() == 0)
        except Exception:   # noqa: BLE001 — best effort; never break the run over profiler control
            started = False
    elif not _warned:
        print("  [cuda_profiler] libcudart not found — profiler_range() is a no-op (CPU / no CUDA).")
        _warned = True
    try:
        yield
    finally:
        if started:
            _CUDART.cudaProfilerStop()
