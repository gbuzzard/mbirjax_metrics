"""
experiments/profiling/region_attribution.py
──────────────────────────────────────────────
Join a Perfetto trace (fusion -> self-time) with the compiled HLO (fusion -> jax.named_scope
region) to get **self-time per code-localized region**.

Why this exists: jax.named_scope tags the HLO op metadata with a stable, code-localized region
name (e.g. ``cone/forward/vertical_fan``) — but the raw trace event names stay XLA-named
(``broadcast_multiply_fusion`` ...) and rename across jax versions.  The HLO is the bridge: each
fusion carries exactly ONE region (validated), so we map trace fusion self-time through the HLO.

FLEXIBLE BY DESIGN: region names are DISCOVERED from the HLO scopes — nothing here hardcodes a
region taxonomy, so adding/renaming named_scopes in the library needs no change here.  A trace
fusion with no HLO region (sharding/orchestration glue, host frames excluded) falls to '(unmapped)'.
"""
import re

from trace_utils import fusion_self_time, is_host_runtime

# A scope path looks like ``<root>/<seg>/<seg>...`` in the HLO op_name metadata.  We accept ANY
# such path (no fixed vocabulary); the default roots are the geometry prefixes we annotate.
_DEFAULT_ROOTS = ("cone", "parallel", "translation", "multiaxis", "qggmrf", "projector")

# jax APPENDS its primitive/transform after the named_scope (e.g. the authored scope
# ``cone/forward/horizontal_fan`` shows in op_name as ``.../horizontal_fan/scatter`` or
# ``.../vertical_fan/vmap/while/body/...``).  To collapse all of a scope's ops back to the AUTHORED
# region we truncate the path at its phase LEAF — the deepest segment WE named.  This is the only
# place tied to the scope vocabulary: it is a tiny set of leaf words, and a new kind of leaf scope
# just needs its word added here (reused phase words like vertical_fan need no change).  Everything
# above (roots, intermediate segments band/pixel/...) is still discovered freely from the HLO.
_PHASE_LEAVES = {"vertical_fan", "horizontal_fan", "assemble", "coord_math", "view_reduce"}


def _to_region(scope_path):
    """Truncate a captured ``cone/.../<phase>/<jax primitive...>`` op-path to the authored region
    (``cone/.../<phase>``).  Cuts at the first PHASE_LEAVES segment; falls back to dropping the
    single trailing primitive segment if no known leaf is present (so a new scope still groups
    sanely, just one level finer)."""
    segs = scope_path.split("/")
    for i, s in enumerate(segs):
        if s in _PHASE_LEAVES:
            return "/".join(segs[:i + 1])
    return "/".join(segs[:-1]) if len(segs) > 1 else scope_path


def _base_name(fusion_name):
    """Strip a trailing instance suffix so trace kernel names join to the HLO fusion base name on
    BOTH backends: XLA:CPU uses ``.N`` (``ynn_fusion.1``) while the GPU/CUPTI kernel names use ``_N``
    (``loop_add_fusion_3``).  Stripping only ``.N`` (the old behavior) left every GPU fusion unmatched
    -> ~68% '(unmapped)'.  Only a trailing dot/underscore + digits is removed (an instance index);
    real names never end that way."""
    return re.sub(r'[._]\d+$', '', fusion_name)


def hlo_fusion_scopes(hlo_text, roots=_DEFAULT_ROOTS):
    """Map each XLA fusion base-name -> ``{region: n_HLO_instances}`` from the HLO op_name metadata.

    Most bases fall in exactly ONE region.  A base in >1 region is a COLLISION: ``_base_name`` strips
    the per-instance suffix to join the trace to the HLO, but that same stripping makes two instances
    in DIFFERENT scopes (e.g. an accumulate ``loop_add_fusion`` in both horizontal_fan and vertical_fan
    on GPU) share one base name — so their trace self-time can't be split apart and is lumped into one
    region.  A base seen only OUTSIDE any scope maps to an EMPTY dict (region None / '(unmapped)').
    Trace fusion-instance suffixes don't line up with HLO ones (CUDA-graph capture renames kernels;
    indices don't correspond), so full-name matching can't disambiguate — hence we SURFACE collisions
    (find_collisions) rather than silently trust the lump.
    """
    root_alt = "|".join(re.escape(r) for r in roots)
    scope_re = re.compile(rf'((?:{root_alt})(?:/[A-Za-z0-9_]+)+)')
    fusion_re = re.compile(r'%([A-Za-z0-9_.\-]*fusion)[.\d]* = .*?op_name="([^"]*)"')
    out = {}
    for line in hlo_text.splitlines():
        m = fusion_re.search(line)
        if not m:
            continue
        counts = out.setdefault(_base_name(m.group(1)), {})
        scope = scope_re.search(m.group(2))
        if scope:
            region = _to_region(scope.group(1))
            counts[region] = counts.get(region, 0) + 1
    return out


def hlo_fusion_regions(hlo_text, roots=_DEFAULT_ROOTS):
    """Map each XLA fusion base-name -> its named_scope region, read from HLO op_name metadata.

    Returns ``{fusion_base_name: region_or_None}``; fusions with no scope map to None.  When a base
    spans >1 region (a COLLISION, see hlo_fusion_scopes) it is assigned to the region holding the MOST
    HLO instances (majority; ties -> first seen) — the best single guess — and find_collisions reports
    that the assigned region's number is then uncertain by up to the colliding base's self-time.
    """
    return {base: (max(counts, key=counts.get) if counts else None)
            for base, counts in hlo_fusion_scopes(hlo_text, roots).items()}


def attribute_regions(trace_path, hlo_text, roots=_DEFAULT_ROOTS):
    """Self-time (microseconds) per named region, joining the trace with the HLO.

    Host/runtime wrapper events (dispatch/wait/threads) are excluded — regions are COMPUTE.  XLA
    fusions with no HLO region go to '(unmapped)'.  Returns ``{region: self_us}`` (descending).
    """
    f2r = hlo_fusion_regions(hlo_text, roots)
    events, _tracks, _n = fusion_self_time(trace_path)
    region_us = {}
    for name, (us, _cnt) in events.items():
        if is_host_runtime(name):
            continue
        region = f2r.get(_base_name(name)) or "(unmapped)"
        region_us[region] = region_us.get(region, 0.0) + us
    return dict(sorted(region_us.items(), key=lambda kv: -kv[1]))


def region_breakdown(trace_path, hlo_text, roots=_DEFAULT_ROOTS):
    """Region breakdown ready for the schema: ``{region: {self_ms, pct}}`` (pct of attributed compute).

    pct is each region's share of the TOTAL attributed compute self-time (regions sum to ~100%),
    a stable, before/after-diffable quantity independent of host-wait noise; absolute self_ms is
    kept alongside so a real speedup is visible too.
    """
    region_us = attribute_regions(trace_path, hlo_text, roots)
    total = sum(region_us.values()) or 1.0
    return {r: {"self_ms": round(us / 1e3, 3), "pct": round(100.0 * us / total, 1)}
            for r, us in region_us.items()}


def find_collisions(trace_path, hlo_text, roots=_DEFAULT_ROOTS):
    """Fusion base-names whose HLO instances span >1 region (see hlo_fusion_scopes) — the cases where
    the base-name join CAN'T cleanly split self-time, so the assigned region's number is uncertain.

    Returns ``[{base, pct, assigned, scopes}]`` sorted by pct desc (pct = the base's share of total
    attributed compute, same denominator as region_breakdown's pct; ``scopes`` = {region: n_instances},
    ``assigned`` = the region the join credited it to).  Empty list = clean attribution.  This SURFACES
    the limitation so a collided region's number is never silently trusted; the caller can scale pct by
    wall to a ms 'amount at stake' alongside the region ms it shares units with.
    """
    scopes = hlo_fusion_scopes(hlo_text, roots)
    assigned = hlo_fusion_regions(hlo_text, roots)
    events, _tracks, _n = fusion_self_time(trace_path)
    base_us = {}
    for name, (us, _cnt) in events.items():
        if is_host_runtime(name):
            continue
        b = _base_name(name)
        base_us[b] = base_us.get(b, 0.0) + us
    total = sum(base_us.values()) or 1.0
    out = [{"base": base, "pct": round(100.0 * base_us.get(base, 0.0) / total, 1),
            "assigned": assigned[base], "scopes": dict(counts)}
           for base, counts in scopes.items() if len(counts) > 1]
    return sorted(out, key=lambda d: -d["pct"])
