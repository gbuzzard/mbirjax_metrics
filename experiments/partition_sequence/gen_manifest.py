"""Write MANIFEST.md into the shared cache dir: a menu of the preprocessed caches so
teammates can see what is already built (and reuse it) without loading any file.

Reads `cache_dir` from config.yaml, scans <tag>.h5 (+ .json sidecar) there, and emits a
markdown table (tag, sinogram shape, downsampling, source, provenance).  Run on the cluster
(the cache dir is on /depot):  python gen_manifest.py
"""
import glob
import json
import os

import h5py

import ps_config

CFG = ps_config.load()
CACHE_DIR = CFG['cache_dir']


def main():
    rows = []
    for h5_path in sorted(glob.glob(os.path.join(CACHE_DIR, '*.h5'))):
        tag = os.path.splitext(os.path.basename(h5_path))[0]
        with h5py.File(h5_path, 'r') as f:
            shape = tuple(f['sinogram'].shape) if 'sinogram' in f else None
        side = {}
        side_path = os.path.join(CACHE_DIR, f'{tag}.json')
        if os.path.exists(side_path):
            side = json.load(open(side_path))
        prov = side.get('provenance', {})
        df, vf = prov.get('detector_factor', '?'), prov.get('view_factor', '?')
        src = os.path.basename(str(prov.get('source', '?')).rstrip('/')) or prov.get('source', '?')
        aligned = 'aligned' if prov.get('aligned') else ''
        rs = side.get('recon_settings', {})
        rs_txt = ', '.join(f'{k}={v}' for k, v in rs.items())
        rows.append((tag, shape, f'v{vf}x d{df}x', src, prov.get('built', '?'),
                     (aligned + ('; ' if aligned and rs_txt else '') + rs_txt)))

    lines = [
        '# Partition-sequence cache manifest',
        '',
        f'Shared, group-writable preprocessed sinograms for the partition-sequence study.',
        f'Location: `{CACHE_DIR}`.  Built by `build_cache.py` (reuses existing caches).',
        'Tag scheme: `<source>_v<view_factor>x_d<det_factor>x_nv<views>_nch<channels>`.  '
        'Regenerate with `python gen_manifest.py`.',
        '',
        '| tag | sinogram (v×row×ch) | subsample | source | built | notes |',
        '|-----|---------------------|-----------|--------|-------|-------|',
    ]
    for tag, shape, dv, src, built, notes in rows:
        shp = '×'.join(str(x) for x in shape) if shape else '?'
        lines.append(f'| `{tag}` | {shp} | {dv} | {src} | {built} | {notes} |')
    lines.append('')

    out = os.path.join(CACHE_DIR, 'MANIFEST.md')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f'wrote {out} ({len(rows)} caches)')


if __name__ == '__main__':
    main()
