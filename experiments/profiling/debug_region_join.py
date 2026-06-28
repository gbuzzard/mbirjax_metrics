"""
experiments/profiling/debug_region_join.py
─────────────────────────────────────────────
One-off diagnostic for "why does a region read ~0%?" — runs the back PIXEL driver (the GPU n=1 path),
joins trace⋈HLO, and prints enough to tell apart the two causes:

  (A) base-name COLLISION  — a fusion base name appears under >1 scope in the HLO, so the join (which
      keys on base name) lumps both into one scope.  -> shows up as a 'COLLISION' line below.
  (B) cross-scope FUSION   — XLA fused a small producer (horizontal_fan, run once) INTO the kernel that
      consumes it (vertical_fan), so horizontal_fan's HLO fusions don't appear in the trace at all.
      -> the region's HLO base names will be MISSING from the trace (in_trace=False).

No CLI args.  Run on the box you're diagnosing:  python experiments/profiling/debug_region_join.py
"""
import os
import sys
import re
import glob
import tempfile
import collections

GEOMETRY = "cone"
SIZE = None   # None -> per-platform profiling size (size_for, matches profile_measure); set a tuple to override

os.environ.setdefault("MBIRJAX_NUM_CPU_DEVICES", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "tooling", "scaling_tests")))
sys.path.insert(0, _HERE)
from profiling_config import size_for   # noqa: E402  (sets MBIRJAX_NUM_CPU_DEVICES; import before jax)
from trace_utils import fusion_self_time, is_host_runtime   # noqa: E402
from region_attribution import hlo_fusion_regions, _base_name, _to_region  # noqa: E402

import mbirjax            # noqa: E402,F401
import jax                # noqa: E402
import performance_tracking as pt   # noqa: E402


def main():
    plat = jax.devices()[0].platform
    size = SIZE or size_for(plat)                            # match profile_measure's per-platform cell
    m = pt.make_model(pt.Config(), GEOMETRY, size)
    if hasattr(m, "configure_devices"):
        m.configure_devices(jax.devices()[:1])
    dev = jax.devices()[0]
    idx = jax.device_put(pt.make_indices(m), dev)
    sino = jax.device_put(pt.make_sinogram(pt.Config(), size), dev)
    pf = m.projector_functions
    run = lambda: pf.sparse_back_project(sino, idx)          # PIXEL driver (GPU n=1 back path)
    print(f"=== back PIXEL driver | {GEOMETRY} {size} | {plat} | jax {jax.__version__} ===")

    for _ in range(2):
        jax.block_until_ready(run())
    td = tempfile.mkdtemp()
    with jax.profiler.trace(td, create_perfetto_trace=True):
        for _ in range(2):
            jax.block_until_ready(run())
    tp = glob.glob(td + "/**/*perfetto*.json.gz", recursive=True)[0]
    hlo = jax.jit(lambda s, p: pf.sparse_back_project(s, p)).lower(sino, idx).compile().as_text()

    # (A) HLO: base -> set of scopes  (collision detector)
    base2scopes = collections.defaultdict(set)
    for line in hlo.splitlines():
        mm = re.search(r'%([A-Za-z0-9_.\-]*fusion)[.\d]* = .*?op_name="([^"]*)"', line)
        if mm:
            s = re.search(r'(cone/[a-z_/]+)', mm.group(2))
            if s:
                base2scopes[_base_name(mm.group(1))].add(_to_region(s.group(1)))
    collisions = {b: sc for b, sc in base2scopes.items() if len(sc) > 1}
    print(f"\n(A) HLO base names: {len(base2scopes)}   COLLISIONS (same base, >1 scope): {len(collisions)}")
    for b, sc in collisions.items():
        print(f"      COLLISION  {b:34} -> {sorted(sc)}")

    # trace base -> self_us
    ev, _t, _n = fusion_self_time(tp)
    trace_us = collections.defaultdict(float)
    for name, (us, _c) in ev.items():
        if not is_host_runtime(name):
            trace_us[_base_name(name)] += us

    # (C) collision detail: for each colliding base, the HLO FULL op names + scopes, and the TRACE
    # FULL names + times.  If trace '<base>_N' lines up with an HLO '<base>.N' (or '_N') of a single
    # scope, then matching on the separator-normalized FULL name (keeping the index) disambiguates.
    if collisions:
        hlo_full = []   # (full_op_name, scope)
        for line in hlo.splitlines():
            mm = re.search(r'%([A-Za-z0-9_.\-]+)\s*=.*?\bfusion\(.*?op_name="([^"]*)"', line)
            if mm:
                s = re.search(r'(cone/[a-z_/]+)', mm.group(2))
                if s:
                    hlo_full.append((mm.group(1), _to_region(s.group(1))))
        for base in collisions:
            print(f"\n(C) collision detail for base '{base}':")
            print("    HLO full op names -> scope:")
            for full, sc in sorted(hlo_full):
                if _base_name(full) == base:
                    print(f"        {full:36} -> {sc}")
            print("    TRACE full names -> self_ms  (xN = trace event count; the band loop executes"
                  " many times, the horizontal accumulate once):")
            for name, (us, c) in sorted(ev.items(), key=lambda kv: -kv[1][0]):
                if not is_host_runtime(name) and _base_name(name) == base:
                    print(f"        {name:36}    {us/1e3:8.2f} ms   x{c}")

    # (B) per region: HLO bases, and whether each is present in the trace + its time
    f2r = hlo_fusion_regions(hlo)
    region_bases = collections.defaultdict(list)
    for b, r in f2r.items():
        region_bases[r or "(unscoped)"].append(b)
    print("\n(B) per region — HLO base names, in_trace?, trace self_ms:")
    for region in sorted(region_bases):
        tot = sum(trace_us.get(b, 0.0) for b in region_bases[region])
        print(f"  {region}   (total {tot/1e3:.1f} ms across {len(region_bases[region])} bases)")
        for b in sorted(region_bases[region], key=lambda x: -trace_us.get(x, 0.0)):
            print(f"      in_trace={b in trace_us!s:5}  {trace_us.get(b,0.0)/1e3:8.2f} ms   {b}")


if __name__ == "__main__":
    main()
